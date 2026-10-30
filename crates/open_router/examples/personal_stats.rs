//! Fetches and validates the personal cache statistics of the OpenRouter
//! provider. Needs a management API key (regular inference keys get a 403):
//! create one, run this example, then delete the key.
//!
//! Usage:
//!   cargo run -p open_router --example personal_stats -- fetch \
//!       [--path <data.jsonc>] [--from DATE] [--to DATE]
//!   cargo run -p open_router --example personal_stats -- validate \
//!       [--path <data.jsonc>] [--from DATE] [--to DATE]
//!
//! DATE is a UTC date of the form YYYY-MM-DD. Both dates are inclusive.
//! --path is the data file: fetch writes it, validate reads it as the
//! baseline. It defaults to the file that the OpenRouter provider embeds.
//! fetch's window defaults to the last 90 days. validate compares the
//! predictions of the baseline against the usage of a later window, which
//! defaults to the usage since the baseline file was written.

use std::{
    collections::HashMap,
    fs,
    io::IsTerminal as _,
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::{Context as _, Result, anyhow, bail};
use clap::{Parser, Subcommand};
use http_client::HttpClient;
use language_model_core::UpstreamProviderPricing;
use open_router::{
    FrontendEndpoint, FrontendPricing, PersonalCacheStats, fetch_usage_measurements,
    list_frontend_endpoints,
};
use reqwest_client::ReqwestClient;
use time::{Date, Duration, OffsetDateTime, format_description::well_known::Rfc3339};

/// The data file that the OpenRouter provider embeds, relative to the crate
/// directory.
const DATA_FILE: &str = "data/personal_cache_stats.jsonc";

/// How far back the fetch window reaches when --from is not given.
const FETCH_WINDOW_DAYS: i64 = 90;

/// Pairs below this prompt volume are too small to score: single sessions
/// dominate their per-token costs.
const VALIDATE_MIN_PROMPT_TOKENS: u64 = 1_000_000;

/// The width at which `#` comment paragraphs wrap.
const COMMENT_WIDTH: usize = 100;

/// Prints a `#` comment paragraph, word-wrapped and gray when stdout is a
/// terminal.
macro_rules! comment {
    ($($arguments:tt)*) => {
        print_comment(&format!($($arguments)*))
    };
}

/// Prints a table heading, bold and gray when stdout is a terminal.
macro_rules! heading {
    ($($arguments:tt)*) => {
        if std::io::stdout().is_terminal() {
            println!("\x1b[1;90m# {}\x1b[0m", format_args!($($arguments)*))
        } else {
            println!("# {}", format_args!($($arguments)*))
        }
    };
}

/// Prints a `#` summary line after a table, yellow when stdout is a terminal.
macro_rules! summary {
    ($($arguments:tt)*) => {
        if std::io::stdout().is_terminal() {
            println!("\x1b[33m# {}\x1b[0m", format_args!($($arguments)*))
        } else {
            println!("# {}", format_args!($($arguments)*))
        }
    };
}

#[derive(Parser)]
struct Args {
    /// The data file: written by fetch, read by validate. Defaults to the
    /// file that the OpenRouter provider embeds
    #[arg(long, global = true)]
    path: Option<PathBuf>,

    /// The first day of the window, in UTC (YYYY-MM-DD)
    #[arg(long, global = true)]
    from: Option<String>,

    /// The last day of the window, in UTC (YYYY-MM-DD)
    #[arg(long, global = true)]
    to: Option<String>,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Fetch personal cache statistics and write the data file
    Fetch,
    /// Validate the data file's predictions against later usage
    Validate,
}

impl Args {
    /// The window bounds from --from/--to: the optional start midnight and
    /// the end instant, which is the midnight after the inclusive --to date,
    /// or now.
    fn window_bounds(&self) -> Result<(Option<OffsetDateTime>, OffsetDateTime)> {
        let from = self
            .from
            .as_deref()
            .map(Self::parse_utc_date)
            .transpose()
            .context("invalid --from")?;
        let to = self
            .to
            .as_deref()
            .map(Self::parse_utc_date)
            .transpose()
            .context("invalid --to")?;
        let end = to
            .map(|date| date.midnight().assume_utc() + Duration::days(1))
            .unwrap_or_else(OffsetDateTime::now_utc);
        let start = from.map(|date| date.midnight().assume_utc());
        Ok((start, end))
    }

    /// The data file path: --path when given, otherwise the file that the
    /// OpenRouter provider embeds.
    fn data_path(&self) -> PathBuf {
        self.path
            .clone()
            .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join(DATA_FILE))
    }

    /// The management API key from the environment, or from a terminal
    /// prompt.
    fn management_key() -> Result<String> {
        if let Ok(key) = std::env::var("OPENROUTER_MANAGEMENT_KEY")
            && !key.is_empty()
        {
            return Ok(key);
        }
        rpassword::prompt_password("OpenRouter management API key: ")
            .map(|key| key.trim().to_owned())
            .context("reading the management key")
    }

    /// Parses a UTC date of the form YYYY-MM-DD.
    fn parse_utc_date(value: &str) -> Result<Date> {
        Date::parse(
            value,
            time::macros::format_description!("[year]-[month]-[day]"),
        )
        .context("expected a UTC date of the form YYYY-MM-DD")
    }

    fn client() -> Result<Arc<dyn HttpClient>> {
        Ok(Arc::new(ReqwestClient::user_agent(concat!(
            env!("CARGO_PKG_NAME"),
            "/",
            env!("CARGO_PKG_VERSION")
        ))?))
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    match &args.command {
        Command::Fetch => fetch(&args),
        Command::Validate => validate(&args),
    }
}

/// The fetch subcommand: fetches the usage measurements and writes the
/// data file that the OpenRouter provider embeds.
fn fetch(args: &Args) -> Result<()> {
    let (start, end) = args.window_bounds()?;
    let start = start.unwrap_or_else(|| end - Duration::days(FETCH_WINDOW_DAYS));
    if start >= end {
        bail!("the window is empty");
    }

    let api_key = Args::management_key()?;
    let client = Args::client()?;
    let stats = reqwest_client::runtime()
        .block_on(fetch_usage_measurements(
            client.as_ref(),
            &api_key,
            start,
            end,
        ))
        .map_err(|error| anyhow!("fetching the usage rows failed: {error:?}"))
        .context("fetching the usage rows")?;

    // The JSONC data file: a header comment with the provenance followed
    // by the serialized statistics. `serde_json_lenient` parses this format.
    let timestamp = |value: OffsetDateTime| value.format(&Rfc3339).context("formatting a time");
    let mut content = String::new();
    content.push_str("// Personal cache statistics for the OpenRouter provider, fetched from\n");
    content.push_str("// OpenRouter's analytics API by the personal_stats example.\n");
    content.push_str("// One row per model and provider, aggregated over the window.\n");
    content.push_str(&format!(
        "// Window: {} to {}, fetched {}.\n",
        timestamp(start)?,
        timestamp(end)?,
        timestamp(OffsetDateTime::now_utc())?
    ));
    content.push_str(&serde_json::to_string_pretty(&stats).context("serializing the statistics")?);
    content.push('\n');
    let data_path = args.data_path();
    fs::write(&data_path, content).context("writing the data file")?;

    println!(
        "Wrote {} measurements to {}",
        stats.len(),
        data_path.display()
    );
    Ok(())
}

/// The validate subcommand: compares the predictions of a baseline data file
/// against the usage of a later window.
fn validate(args: &Args) -> Result<()> {
    let baseline_path = args.data_path();
    let (start, end) = args.window_bounds()?;
    let default_start: OffsetDateTime = fs::metadata(&baseline_path)
        .context("reading the baseline data file")?
        .modified()
        .context("reading the baseline modification time")?
        .try_into()
        .context("converting the baseline modification time")?;
    let start = start.unwrap_or(default_start);
    if start >= end {
        bail!("the validation window is empty");
    }
    let baseline = PersonalCacheStats::from_jsonc(
        &fs::read_to_string(&baseline_path).context("reading the baseline data file")?,
    );

    let client = Args::client()?;
    let runtime = reqwest_client::runtime();

    let api_key = Args::management_key()?;
    let validation = runtime
        .block_on(fetch_usage_measurements(
            client.as_ref(),
            &api_key,
            start,
            end,
        ))
        .map_err(|error| anyhow!("fetching the validation usage failed: {error:?}"))
        .context("fetching the validation usage")?;

    let mut frontend: HashMap<String, HashMap<String, FrontendEndpoint>> = HashMap::new();
    let mut unavailable_models = Vec::new();
    for model in validation.models() {
        let model = model.to_string();
        match runtime.block_on(list_frontend_endpoints(client.as_ref(), &model)) {
            Ok(endpoints) => {
                frontend.insert(model.clone(), endpoints);
            }
            Err(error) => {
                eprintln!("no website data for {model}: {error:?}");
                unavailable_models.push(model);
            }
        }
    }

    heading!("Analysis Window");
    comment!(
        "Showing results for usage data from {} to {}",
        start.format(&Rfc3339)?,
        end.format(&Rfc3339)?
    );
    if !unavailable_models.is_empty() {
        summary!(
            "Models without website data: {}",
            unavailable_models.join(", ")
        );
    }

    let mut rate_diffs = Vec::new();
    println!();
    heading!("Cache hit rate");
    comment!(
        "Shows the predicted cache rate (from the baseline) against the actual cache in the validation window."
    );
    let mut rates = Table::new(
        &[
            "Model",
            "Provider",
            "Predicted",
            "Actual",
            "Delta",
            "Token Volume",
        ],
        &[false, false, true, true, true, true],
    );
    for model in validation.models() {
        let Some(endpoints) = frontend.get(model) else {
            continue;
        };
        for (slug, endpoint) in endpoints {
            let Some(provider) = endpoint.provider_name.as_deref() else {
                continue;
            };
            let Some(actual) = validation.cache_hit_rate(model, provider) else {
                continue;
            };
            let prompt = validation.tokens_prompt(model, provider);
            if prompt < VALIDATE_MIN_PROMPT_TOKENS {
                continue;
            }
            let Some(predicted) = baseline.calibrated_cache_hit_rate(model, endpoint, endpoints)
            else {
                continue;
            };
            let diff_pp = (actual - predicted) * 100.0;
            rates.push(
                prompt as f64,
                vec![
                    model.to_string(),
                    provider_label(provider, slug),
                    format!("{:.1}%", predicted * 100.0),
                    format!("{:.1}%", actual * 100.0),
                    sign_color(diff_pp, |value| format!("{value:+.1}%")),
                    format!("{:.2} M", prompt as f64 / 1e6),
                ],
            );
            rate_diffs.push(diff_pp.abs());
        }
    }
    rates.print();
    summary!("Median |Delta|: {}", median_line(&mut rate_diffs, 1));

    let mut q_diffs = Vec::new();
    println!();
    heading!("Output-to-input ratio");
    comment!(
        "Predicted ratio of output tokens over input tokens, vs. measured ratio in the validation window."
    );
    let mut ratios = Table::new(
        &["Model", "Predicted", "Actual", "Delta", "Token Volume"],
        &[false, true, true, true, true],
    );
    for model in validation.models() {
        let Some(predicted) = baseline.output_input_ratio(model) else {
            continue;
        };
        let Some(actual) = validation.output_input_ratio(model) else {
            continue;
        };
        let diff_pct = (actual - predicted) / predicted * 100.0;
        let prompt = validation.tokens_prompt_total(model);
        ratios.push(
            prompt as f64,
            vec![
                model.to_string(),
                format!("{predicted:.4}"),
                format!("{actual:.4}"),
                sign_color(diff_pct, |value| format!("{value:+.1}%")),
                format!("{:.2} M", prompt as f64 / 1e6),
            ],
        );
        q_diffs.push(diff_pct.abs());
    }
    ratios.print();
    summary!("Median |Delta|: {}", median_line(&mut q_diffs, 1));

    // The summaries weight each row by its prompt tokens; providers with
    // several endpoints contribute once per endpoint.
    let mut weight_sum = 0.0;
    let mut weighted_err = 0.0;
    let mut weighted_residual = 0.0;
    let mut total_predicted = 0.0;
    let mut total_actual = 0.0;
    let mut total_reported = 0.0;
    println!();
    heading!("Total cost per endpoint");
    comment!(
        "Shows the total cost of the traffic in the validation window. Total
        cost is shown with predicted cache rates vs actual cache rates and vs.
        final OpenRouter prices. OpenRouter prices include pricing changes and
        discounts in the validation window."
    );
    let mut costs = Table::new(
        &[
            "Model",
            "Provider",
            "Predicted",
            "Actual Cache Rates",
            "OpenRouter",
        ],
        &[false, false, true, true, true],
    );
    for model in validation.models() {
        let Some(endpoints) = frontend.get(model) else {
            continue;
        };
        let Some(q_baseline) = baseline.output_input_ratio(model) else {
            continue;
        };
        let q_actual = validation.output_input_ratio(model);
        for (slug, endpoint) in endpoints {
            let Some(provider) = endpoint.provider_name.as_deref() else {
                continue;
            };
            let Some(pricing) = endpoint.pricing.as_ref() else {
                continue;
            };
            let Some(actual_usage) = validation.total_usage(model, provider) else {
                continue;
            };
            let prompt = validation.tokens_prompt(model, provider);
            if prompt < VALIDATE_MIN_PROMPT_TOKENS {
                continue;
            }
            let Some(actual_rate) = validation.cache_hit_rate(model, provider) else {
                continue;
            };
            let Some(predicted_rate) =
                baseline.calibrated_cache_hit_rate(model, endpoint, endpoints)
            else {
                continue;
            };
            let predicted = blended(pricing, predicted_rate, q_baseline);
            let actual = blended(pricing, actual_rate, q_actual.unwrap_or(q_baseline));
            let predicted_total = predicted * prompt as f64 / 1e6;
            let actual_total = actual * prompt as f64 / 1e6;
            let actual_diff = sign_color(-(actual_total - predicted_total), |value| {
                let value = -value;
                format!("{value:+.2}")
            });
            let usage_diff = sign_color(-(actual_usage - predicted_total), |value| {
                let value = -value;
                format!("{value:+.2}")
            });
            costs.push(
                actual_usage,
                vec![
                    model.to_string(),
                    provider_label(provider, slug),
                    format!("${predicted_total:.2}"),
                    format!("${actual_total:.2} ({actual_diff})"),
                    format!("${actual_usage:.2} ({usage_diff})"),
                ],
            );
            // The valuation uses the current prices too, so what remains in
            // the residual is price movement and the price difference of
            // endpoints that were not used.
            if actual_usage > 0.0 {
                let weight = prompt as f64;
                weighted_err += (predicted_total - actual_usage).abs() / actual_usage * weight;
                weighted_residual += (actual_total - actual_usage).abs() / actual_usage * weight;
                weight_sum += weight;
            }
            total_predicted += predicted_total;
            total_actual += actual_total;
            total_reported += actual_usage;
        }
    }
    costs.print();
    if weight_sum > 0.0 {
        summary!(
            "Volume-weighted mean |err|: {:.1}% of the reported price",
            100.0 * weighted_err / weight_sum,
        );
        summary!(
            "Volume-weighted mean |err| from price movement and endpoint prices: {:.1}%",
            100.0 * weighted_residual / weight_sum,
        );
    }
    summary!(
        "Totals: Predicted ${total_predicted:.2}, Actual rates ${total_actual:.2} ({:+.2}), reported ${total_reported:.2} (predicted {:+.2})",
        total_actual - total_predicted,
        total_reported - total_predicted,
    );

    Ok(())
}

/// The estimated cost per million input tokens at `rate` and
/// `output_input_ratio`, from the endpoint's per-token prices.
fn blended(pricing: &FrontendPricing, rate: f64, output_input_ratio: f64) -> f64 {
    let mut pricing = UpstreamProviderPricing::from(pricing);
    pricing.cache_hit_rate = Some(rate);
    pricing.output_input_ratio = output_input_ratio;
    pricing.blended()
}

/// The provider name with the endpoint variant in parentheses: the endpoint
/// slug without its leading provider component. The variant is light gray
/// when stdout is a terminal.
fn provider_label(provider: &str, slug: &str) -> String {
    match slug.split_once('/') {
        Some((_, variant)) if !variant.is_empty() => {
            if std::io::stdout().is_terminal() {
                format!("{provider} \x1b[90m({variant})\x1b[0m")
            } else {
                format!("{provider} ({variant})")
            }
        }
        _ => provider.to_owned(),
    }
}

/// Colors the text formatted from `value` by its sign when stdout is a
/// terminal: green for positive, red for negative.
fn sign_color(value: f64, format_text: impl FnOnce(f64) -> String) -> String {
    if !std::io::stdout().is_terminal() || value == 0.0 {
        return format_text(value);
    }
    let color = if value > 0.0 { "\x1b[32m" } else { "\x1b[31m" };
    format!("{color}{}\x1b[0m", format_text(value))
}

/// Prints `text` as a word-wrapped `#` comment paragraph, gray when stdout
/// is a terminal.
fn print_comment(text: &str) {
    let color = std::io::stdout().is_terminal();
    for line in wrap(text, COMMENT_WIDTH) {
        if color {
            println!("\x1b[90m# {line}\x1b[0m");
        } else {
            println!("# {line}");
        }
    }
}

/// Splits `text` into lines of at most `width` characters, breaking on
/// whitespace. Words longer than `width` keep their own line.
fn wrap(text: &str, width: usize) -> Vec<String> {
    let mut lines = Vec::new();
    let mut line = String::new();
    for word in text.split_whitespace() {
        if !line.is_empty() && line.chars().count() + 1 + word.chars().count() > width {
            lines.push(std::mem::take(&mut line));
        }
        if !line.is_empty() {
            line.push(' ');
        }
        line.push_str(word);
    }
    if !line.is_empty() {
        lines.push(line);
    }
    lines
}

fn median_line(values: &mut [f64], decimals: usize) -> String {
    if values.is_empty() {
        return "n/a".to_string();
    }
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    let median = if values.len() % 2 == 1 {
        values[middle]
    } else {
        (values[middle - 1] + values[middle]) / 2.0
    };
    format!("{:.*}", decimals, median)
}

/// A report table that buffers rows and prints columns sized to the widest
/// cell, with an underlined header row and a rule at the bottom. Rows print
/// largest `sort_key` first, ties in cell order.
struct Table {
    headers: Vec<String>,
    /// Whether each column holds numbers and is right-aligned.
    right: Vec<bool>,
    rows: Vec<(f64, Vec<String>)>,
}

impl Table {
    /// Creates a table whose columns with `true` in `right` are right-aligned.
    fn new(headers: &[&str], right: &[bool]) -> Self {
        Self {
            headers: headers.iter().map(|header| header.to_string()).collect(),
            right: right.to_vec(),
            rows: Vec::new(),
        }
    }

    fn push(&mut self, sort_key: f64, row: Vec<String>) {
        self.rows.push((sort_key, row));
    }

    /// The number of visible characters in `cell`, ignoring ANSI color
    /// escapes.
    fn visible_len(cell: &str) -> usize {
        let mut length = 0;
        let mut characters = cell.chars();
        while let Some(character) = characters.next() {
            if character == '\x1b' {
                for character in characters.by_ref() {
                    if character == 'm' {
                        break;
                    }
                }
            } else {
                length += 1;
            }
        }
        length
    }

    fn print(&self) {
        let color = std::io::stdout().is_terminal();
        let mut widths: Vec<usize> = self
            .headers
            .iter()
            .map(|header| Self::visible_len(header))
            .collect();
        for (_, row) in &self.rows {
            for (width, cell) in widths.iter_mut().zip(row) {
                *width = (*width).max(Self::visible_len(cell));
            }
        }
        let line = |cells: &[String], underline: bool| {
            cells
                .iter()
                .enumerate()
                .map(|(column, cell)| {
                    let padding =
                        " ".repeat(widths[column].saturating_sub(Self::visible_len(cell)));
                    let padded = if self.right[column] {
                        format!("{padding}{cell}")
                    } else {
                        format!("{cell}{padding}")
                    };
                    if underline && color {
                        format!("\x1b[4m{padded}\x1b[0m")
                    } else {
                        padded
                    }
                })
                .collect::<Vec<_>>()
                .join("  ")
        };
        println!("{}", line(&self.headers, true));
        let mut rows: Vec<&(f64, Vec<String>)> = self.rows.iter().collect();
        rows.sort_by(|(left_key, left_row), (right_key, right_row)| {
            right_key.total_cmp(left_key).then(left_row.cmp(right_row))
        });
        for (_, row) in rows {
            println!("{}", line(row, false));
        }
        let rule_width = widths.iter().sum::<usize>() + 2 * widths.len().saturating_sub(1);
        let rule = "─".repeat(rule_width);
        if color {
            println!("\x1b[90m{rule}\x1b[0m");
        } else {
            println!("{rule}");
        }
    }
}
