use std::collections::HashMap;
use std::sync::LazyLock;

use http_client::HttpClient;
use serde::{Deserialize, Serialize, Serializer};
use time::OffsetDateTime;

use crate::OpenRouterError;
use crate::analytics::AnalyticsRow;
use crate::frontend_stats::FrontendEndpoint;

/// The prompt-token volume at which a personal measurement carries half the
/// weight of the OpenRouter rate, and below which a provider does not
/// contribute to the offset.
const MEASUREMENT_PRIOR_TOKENS: f64 = 10_000_000.0;

/// The highest plausible cache hit rate estimate. Every thread starts with a
/// cold cache, so a rate of 100% is impossible.
const MAX_CACHE_HIT_RATE: f64 = 0.99;

/// A measurement of the token volume and cached tokens of a (model, provider)
/// pair. The model is the model's canonical slug, the provider is identified by
/// name.
#[derive(Debug, Serialize, Deserialize)]
pub struct PersonalMeasurement {
    pub model: String,
    pub provider: String,
    #[serde(default)]
    pub cached_tokens: u64,
    #[serde(default)]
    pub tokens_prompt: u64,
    #[serde(default)]
    pub tokens_completion: u64,
    /// The cost of the requests in the measurement window, in USD.
    #[serde(default)]
    pub total_usage: Option<f64>,
}

impl PersonalMeasurement {
    /// The share of prompt tokens that were served from the cache.
    fn cache_hit_rate(&self) -> Option<f64> {
        (self.tokens_prompt > 0).then(|| self.cached_tokens as f64 / self.tokens_prompt as f64)
    }

    /// Converts an analytics row into a measurement. Rows without a model
    /// or provider are aggregates that the statistics cannot use.
    fn from_analytics_row(row: AnalyticsRow) -> Option<Self> {
        let model = row.model.filter(|model| !model.is_empty())?;
        let provider = row.provider.filter(|provider| !provider.is_empty())?;
        Some(Self {
            model,
            provider,
            cached_tokens: row.cached_tokens.unwrap_or(0),
            tokens_prompt: row.tokens_prompt.unwrap_or(0),
            tokens_completion: row.tokens_completion.unwrap_or(0),
            total_usage: row.total_usage,
        })
    }
}

/// Fetches the usage rows of this workload from OpenRouter's analytics API
/// and aggregates them into statistics.
pub async fn fetch_usage_measurements(
    client: &dyn HttpClient,
    api_key: &str,
    start: OffsetDateTime,
    end: OffsetDateTime,
) -> Result<PersonalCacheStats, OpenRouterError> {
    let rows: Vec<AnalyticsRow> = crate::analytics::query_analytics(
        client,
        api_key,
        &[
            "cached_tokens",
            "tokens_prompt",
            "tokens_completion",
            "total_usage",
        ],
        &["model", "provider"],
        start,
        end,
    )
    .await?;
    Ok(PersonalCacheStats::from_measurements(
        rows.into_iter()
            .filter_map(PersonalMeasurement::from_analytics_row)
            .collect(),
    ))
}

/// The personal cache statistics of this workload, keyed by model slug and
/// provider name.
#[derive(Debug, Default)]
pub struct PersonalCacheStats(HashMap<String, HashMap<String, PersonalMeasurement>>);

impl PersonalCacheStats {
    /// Loads the statistics from the JSONC data that the `fetch` subcommand
    /// of the `personal_stats` example writes. Panics when the data does not
    /// parse or validate.
    pub fn from_jsonc(json: &str) -> Self {
        let measurements: Vec<PersonalMeasurement> =
            serde_json_lenient::from_str(json).expect("the personal cache stats parse");
        Self::from_measurements(measurements)
    }

    /// Builds the statistics from raw measurement rows. Panics when a row
    /// does not validate.
    pub fn from_measurements(measurements: Vec<PersonalMeasurement>) -> Self {
        validate(&measurements);
        let mut stats = Self::default();
        for row in measurements {
            stats.add_measurement(row);
        }
        stats
    }

    /// Adds a measurement data point for a (model, provider) pair. Measurements
    /// are assumed to be non-overlapping.
    fn add_measurement(&mut self, row: PersonalMeasurement) {
        self.0
            .entry(row.model.clone())
            .or_default()
            .entry(row.provider.clone())
            .and_modify(|measurement| {
                measurement.cached_tokens += row.cached_tokens;
                measurement.tokens_prompt += row.tokens_prompt;
                measurement.tokens_completion += row.tokens_completion;
                measurement.total_usage = match (measurement.total_usage, row.total_usage) {
                    (Some(existing), Some(row_usage)) => Some(existing + row_usage),
                    _ => None,
                };
            })
            .or_insert(row);
    }

    /// The measured statistics of this workload with `provider` serving
    /// `model`, if any.
    fn measurement(&self, model: &str, provider: &str) -> Option<&PersonalMeasurement> {
        self.0.get(model)?.get(provider)
    }

    /// The number of measured (model, provider) pairs.
    pub fn len(&self) -> usize {
        self.0.values().map(|providers| providers.len()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.0.values().all(|providers| providers.is_empty())
    }

    /// The models with measurements in this workload.
    pub fn models(&self) -> impl Iterator<Item = &str> {
        self.0.keys().map(|model| model.as_str())
    }

    /// The providers with measurements for `model`.
    pub fn providers(&self, model: &str) -> impl Iterator<Item = &str> {
        self.0
            .get(model)
            .into_iter()
            .flatten()
            .map(|(provider, _)| provider.as_str())
    }

    /// The measured cache hit rate of this workload with `provider` serving
    /// `model`, if any.
    pub fn cache_hit_rate(&self, model: &str, provider: &str) -> Option<f64> {
        self.measurement(model, provider)?.cache_hit_rate()
    }

    /// The prompt tokens measured for `provider` serving `model`.
    pub fn tokens_prompt(&self, model: &str, provider: &str) -> u64 {
        self.measurement(model, provider)
            .map(|measurement| measurement.tokens_prompt)
            .unwrap_or_default()
    }

    /// The prompt tokens measured for `model` across all providers.
    pub fn tokens_prompt_total(&self, model: &str) -> u64 {
        self.0
            .get(model)
            .into_iter()
            .flatten()
            .map(|(_, measurement)| measurement.tokens_prompt)
            .sum()
    }

    /// The cost of the measured requests with `provider` serving `model`,
    /// when the measurement carries cost data.
    pub fn total_usage(&self, model: &str, provider: &str) -> Option<f64> {
        self.measurement(model, provider)?.total_usage
    }

    /// How much higher (or lower) the measured workload's cache rate is than
    /// the OpenRouter rate. This is computed as the median over the measured
    /// data for `model`. Low-volume providers are excluded.
    fn cache_rate_offset(&self, model: &str, endpoints: &HashMap<String, FrontendEndpoint>) -> f64 {
        let mut diffs: Vec<f64> = endpoints
            .values()
            .filter_map(|endpoint| {
                let measurement = self.measurement(model, endpoint.provider_name.as_deref()?)?;
                let own = measurement.cache_hit_rate()?;
                let global = endpoint.cache_hit_rate?;
                (measurement.tokens_prompt as f64 >= MEASUREMENT_PRIOR_TOKENS).then(|| own - global)
            })
            .collect();
        median(&mut diffs)
    }

    /// The output-to-input token ratio of `model` in this workload, when
    /// measured.
    pub fn output_input_ratio(&self, model: &str) -> Option<f64> {
        let mut prompt = 0u64;
        let mut completion = 0u64;
        for measurement in self.0.get(model)?.values() {
            prompt += measurement.tokens_prompt;
            completion += measurement.tokens_completion;
        }
        (prompt > 0 && completion > 0).then(|| completion as f64 / prompt as f64)
    }

    /// The calibrated cache hit rate for `endpoint` serving `model`, blending
    /// the personal measurement of this workload (if any) with the rate
    /// OpenRouter reports for the endpoint. Returns `None` when OpenRouter
    /// reports no rate for the endpoint.
    pub fn calibrated_cache_hit_rate(
        &self,
        model: &str,
        endpoint: &FrontendEndpoint,
        endpoints: &HashMap<String, FrontendEndpoint>,
    ) -> Option<f64> {
        let prior = (endpoint.cache_hit_rate? + self.cache_rate_offset(model, endpoints))
            .min(MAX_CACHE_HIT_RATE);
        let Some(name) = endpoint.provider_name.as_deref() else {
            return Some(prior);
        };
        let Some(measurement) = self.measurement(model, name) else {
            return Some(prior);
        };
        let Some(own) = measurement.cache_hit_rate() else {
            return Some(prior);
        };
        // Empirical Bayes shrinkage: the prior acts as MEASUREMENT_PRIOR_TOKENS
        // pseudo prompt tokens, so the personal rate carries the weight
        // n / (n + MEASUREMENT_PRIOR_TOKENS).
        let lambda = measurement.tokens_prompt as f64
            / (measurement.tokens_prompt as f64 + MEASUREMENT_PRIOR_TOKENS);
        Some(lambda * own + (1.0 - lambda) * prior)
    }
}

/// Serializes as the flat array of measurements. Sorted by model and provider
/// so that regenerated data files diff cleanly.
impl Serialize for PersonalCacheStats {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut rows: Vec<&PersonalMeasurement> = self
            .0
            .values()
            .flat_map(|providers| providers.values())
            .collect();
        rows.sort_unstable_by(|left, right| {
            (&left.model, &left.provider).cmp(&(&right.model, &right.provider))
        });
        rows.serialize(serializer)
    }
}

/// Personal cache statistics from the data file that the `fetch` subcommand
/// of the `personal_stats` example fetches from OpenRouter's analytics API.
/// Regenerate the data file to refresh the calibration.
static PERSONAL_CACHE_STATS: LazyLock<PersonalCacheStats> = LazyLock::new(|| {
    PersonalCacheStats::from_jsonc(include_str!("../data/personal_cache_stats.jsonc"))
});

/// The personal cache statistics embedded in the binary.
pub fn personal_cache_stats() -> &'static PersonalCacheStats {
    &PERSONAL_CACHE_STATS
}

/// Rejects measurements with more cached tokens than prompt tokens, so that
/// the calibration can rely on bounded rates.
fn validate(measurements: &[PersonalMeasurement]) {
    for measurement in measurements {
        assert!(
            measurement.cached_tokens <= measurement.tokens_prompt,
            "{} cached tokens of {} on {} exceed the {} prompt tokens",
            measurement.cached_tokens,
            measurement.provider,
            measurement.model,
            measurement.tokens_prompt
        );
    }
}

/// The middle value of `values`, or the mean of the two middle values, or
/// zero when `values` is empty.
fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|left, right| left.total_cmp(right));
    let middle = values.len() / 2;
    if values.len() % 2 == 1 {
        values[middle]
    } else {
        (values[middle - 1] + values[middle]) / 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn date(year: i32, month: u8, day: u8) -> OffsetDateTime {
        time::Date::from_calendar_date(year, time::Month::try_from(month).expect("month"), day)
            .expect("date")
            .midnight()
            .assume_utc()
    }

    /// Builds the statistics map from raw rows, the same way the embedded
    /// data is loaded.
    fn stats(measurements: Vec<PersonalMeasurement>) -> PersonalCacheStats {
        let mut stats = PersonalCacheStats::default();
        for row in measurements {
            stats.add_measurement(row);
        }
        stats
    }

    fn measurement(
        model: &str,
        provider: &str,
        cached_tokens: u64,
        tokens_prompt: u64,
        tokens_completion: u64,
    ) -> PersonalMeasurement {
        PersonalMeasurement {
            model: model.to_string(),
            provider: provider.to_string(),
            cached_tokens,
            tokens_prompt,
            tokens_completion,
            total_usage: None,
        }
    }

    fn endpoints(pairs: &[(&str, &str, f64)]) -> HashMap<String, FrontendEndpoint> {
        pairs
            .iter()
            .map(|(tag, name, rate)| {
                (
                    tag.to_string(),
                    FrontendEndpoint {
                        provider_name: Some(name.to_string()),
                        cache_hit_rate: Some(*rate),
                        region: None,
                        service_tier: None,
                        headquarters: None,
                        datacenters: Vec::new(),
                        icon_url: None,
                        pricing: None,
                    },
                )
            })
            .collect()
    }

    #[test]
    fn analytics_rows_convert_to_measurements() {
        let row = AnalyticsRow {
            start: date(2026, 10, 2),
            end: date(2026, 10, 3),
            model: Some("m".to_string()),
            provider: Some("A".to_string()),
            cached_tokens: Some(90),
            tokens_prompt: Some(100),
            tokens_completion: Some(10),
            total_usage: Some(0.5),
        };
        let measurement = PersonalMeasurement::from_analytics_row(row).expect("measurement");
        assert_eq!(measurement.cached_tokens, 90);
        assert_eq!(measurement.tokens_prompt, 100);
        assert_eq!(measurement.total_usage, Some(0.5));

        // Rows without a model or provider are aggregates that the
        // statistics cannot use.
        let row = AnalyticsRow {
            start: date(2026, 10, 2),
            end: date(2026, 10, 3),
            model: Some(String::new()),
            provider: Some("A".to_string()),
            cached_tokens: None,
            tokens_prompt: None,
            tokens_completion: None,
            total_usage: None,
        };
        assert!(PersonalMeasurement::from_analytics_row(row).is_none());
    }

    #[test]
    fn add_measurement_sums_the_token_counts() {
        let mut stats = PersonalCacheStats::default();
        stats.add_measurement(measurement("m", "A", 90, 100, 10));
        stats.add_measurement(measurement("m", "A", 50, 100, 30));

        let merged = stats.measurement("m", "A").unwrap();
        assert!((merged.cache_hit_rate().unwrap() - 0.7).abs() < 1e-9);
        assert_eq!(merged.tokens_prompt, 200);
        assert_eq!(merged.tokens_completion, 40);
    }

    fn measurement_with_cost(
        model: &str,
        provider: &str,
        tokens_prompt: u64,
        tokens_completion: u64,
        total_usage: f64,
    ) -> PersonalMeasurement {
        PersonalMeasurement {
            model: model.to_string(),
            provider: provider.to_string(),
            cached_tokens: 0,
            tokens_prompt,
            tokens_completion,
            total_usage: Some(total_usage),
        }
    }

    #[test]
    fn add_measurement_sums_the_total_usage() {
        let mut stats = PersonalCacheStats::default();
        stats.add_measurement(measurement_with_cost("m", "A", 100, 0, 5.0));
        stats.add_measurement(measurement_with_cost("m", "A", 300, 0, 15.0));

        let merged = stats.measurement("m", "A").unwrap();
        assert_eq!(merged.total_usage, Some(20.0));
    }

    #[test]
    fn a_window_without_cost_data_hides_the_total_usage() {
        let mut stats = PersonalCacheStats::default();
        stats.add_measurement(measurement_with_cost("m", "A", 100, 0, 5.0));
        stats.add_measurement(measurement("m", "A", 0, 300, 0));

        let merged = stats.measurement("m", "A").unwrap();
        assert_eq!(merged.total_usage, None);
    }

    #[test]
    fn add_measurement_keeps_models_and_providers_separate() {
        let mut stats = PersonalCacheStats::default();
        stats.add_measurement(measurement("m", "A", 90, 100, 0));
        stats.add_measurement(measurement("m", "B", 50, 100, 0));
        stats.add_measurement(measurement("other", "A", 10, 100, 0));

        assert!(stats.measurement("m", "A").is_some());
        assert!(stats.measurement("m", "B").is_some());
        assert!(stats.measurement("other", "A").is_some());
        assert!(stats.measurement("other", "B").is_none());
    }

    #[test]
    fn cache_rate_offset_is_the_median_over_measured_endpoints() {
        let stats = stats(vec![measurement("m", "A", 90_000_000, 100_000_000, 0)]);
        let endpoints = endpoints(&[("a/fp8", "A", 0.8), ("a/fp4", "A", 0.7), ("b", "B", 0.6)]);

        // diffs: 0.1 and 0.2 → median 0.15; provider "B" is unmeasured and
        // does not contribute.
        assert!((stats.cache_rate_offset("m", &endpoints) - 0.15).abs() < 1e-9);
    }

    #[test]
    fn measurements_match_by_provider_name() {
        let stats = stats(vec![measurement(
            "m",
            "InferenceNet",
            90_000_000,
            100_000_000,
            0,
        )]);

        assert!(stats.measurement("m", "InferenceNet").is_some());
        // The display name is a different provider as far as the analytics
        // data is concerned.
        assert!(stats.measurement("m", "inference.net").is_none());
    }
}
