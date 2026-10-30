//! The usage analytics of OpenRouter's management API.

use futures::AsyncReadExt;
use http_client::{AsyncBody, HttpClient, Method, Request as HttpRequest};
use serde::{Deserialize, Serialize};
use time::{Duration, OffsetDateTime};

use crate::{
    ApiError, OPEN_ROUTER_API_URL, OpenRouterError, OpenRouterErrorBody, OpenRouterErrorResponse,
    retry_after_with_rate_limit_default,
};

/// The maximum number of days a single analytics query may cover.
const MAX_QUERY_DAYS: i64 = 31;

/// A row of the analytics response. Each field corresponds to a possible metric
/// in the OpenRouter metrics API. A field is absent when its metric was not
/// requested.
#[derive(Debug)]
pub struct AnalyticsRow {
    pub start: OffsetDateTime,
    pub end: OffsetDateTime,
    pub model: Option<String>,
    pub provider: Option<String>,
    pub cached_tokens: Option<u64>,
    pub tokens_prompt: Option<u64>,
    pub tokens_completion: Option<u64>,
    pub total_usage: Option<f64>,
}

/// The rows of the analytics response for `query`. Ranges longer than a
/// single query are covered with consecutive queries.
pub async fn query_analytics(
    client: &dyn HttpClient,
    api_key: &str,
    metrics: &[&str],
    dimensions: &[&str],
    start: OffsetDateTime,
    end: OffsetDateTime,
) -> Result<Vec<AnalyticsRow>, OpenRouterError> {
    let mut rows = Vec::new();
    let mut chunk_start = start;
    while chunk_start < end {
        let chunk_end = (chunk_start + Duration::days(MAX_QUERY_DAYS)).min(end);
        rows.extend(
            query_chunk(client, api_key, metrics, dimensions, chunk_start, chunk_end).await?,
        );
        chunk_start = chunk_end;
    }
    Ok(rows)
}

async fn query_chunk(
    client: &dyn HttpClient,
    api_key: &str,
    metrics: &[&str],
    dimensions: &[&str],
    start: OffsetDateTime,
    end: OffsetDateTime,
) -> Result<Vec<AnalyticsRow>, OpenRouterError> {
    let uri = format!("{OPEN_ROUTER_API_URL}/analytics/query");
    let body = serde_json::to_string(&AnalyticsQueryBody {
        metrics,
        dimensions,
        time_range: TimeRange { start, end },
    })
    .map_err(OpenRouterError::SerializeBody)?;
    let request = HttpRequest::builder()
        .method(Method::POST)
        .uri(uri)
        .header("Accept", "application/json")
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .body(AsyncBody::from(body))
        .map_err(OpenRouterError::BuildRequestBody)?;
    let host = request
        .uri()
        .host()
        .unwrap_or(OPEN_ROUTER_API_URL)
        .to_owned();
    let mut response = client
        .send(request)
        .await
        .map_err(|error| OpenRouterError::HttpSend { host, error })?;

    let mut body = String::new();
    response
        .body_mut()
        .read_to_string(&mut body)
        .await
        .map_err(OpenRouterError::ReadResponse)?;

    if response.status().is_success() {
        let response: AnalyticsResponse =
            serde_json::from_str(&body).map_err(OpenRouterError::DeserializeResponse)?;
        Ok(response
            .data
            .data
            .into_iter()
            .map(|entry| AnalyticsRow {
                start,
                end,
                model: entry.model,
                provider: entry.provider,
                cached_tokens: entry.cached_tokens,
                tokens_prompt: entry.tokens_prompt,
                tokens_completion: entry.tokens_completion,
                total_usage: entry.total_usage,
            })
            .collect())
    } else {
        let status = response.status();
        let error_response = match serde_json::from_str::<OpenRouterErrorResponse>(&body) {
            Ok(OpenRouterErrorResponse { error }) => error,
            Err(_) => OpenRouterErrorBody {
                code: status.as_u16(),
                message: body,
                metadata: None,
            },
        };

        Err(OpenRouterError::ApiError(ApiError {
            status: Some(status.as_u16()),
            code: error_response.code,
            message: error_response.message,
            retry_after: retry_after_with_rate_limit_default(status, response.headers()),
        }))
    }
}

#[derive(Serialize)]
struct AnalyticsQueryBody<'a> {
    metrics: &'a [&'a str],
    dimensions: &'a [&'a str],
    time_range: TimeRange,
}

/// The time range of an analytics query. The end is exclusive.
#[derive(Serialize)]
struct TimeRange {
    #[serde(with = "time::serde::rfc3339")]
    start: OffsetDateTime,
    #[serde(with = "time::serde::rfc3339")]
    end: OffsetDateTime,
}

#[derive(Debug, Deserialize)]
struct AnalyticsResponse {
    data: AnalyticsData,
}

#[derive(Debug, Deserialize)]
struct AnalyticsData {
    #[serde(default)]
    data: Vec<AnalyticsRowEntry>,
}

/// A row of the analytics API response.
#[derive(Debug, Deserialize)]
struct AnalyticsRowEntry {
    model: Option<String>,
    provider: Option<String>,
    #[serde(default, deserialize_with = "deserialize_count")]
    cached_tokens: Option<u64>,
    #[serde(default, deserialize_with = "deserialize_count")]
    tokens_prompt: Option<u64>,
    #[serde(default, deserialize_with = "deserialize_count")]
    tokens_completion: Option<u64>,
    #[serde(default)]
    total_usage: Option<f64>,
}

/// The token counts arrive as decimal strings, but numbers are accepted for
/// robustness.
fn deserialize_count<'de, D>(deserializer: D) -> Result<Option<u64>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = Option::<serde_json::Value>::deserialize(deserializer)?;
    Ok(value.map(count))
}

fn count(value: serde_json::Value) -> u64 {
    match value {
        serde_json::Value::String(value) => value.parse().unwrap_or(0),
        serde_json::Value::Number(number) => number.as_u64().unwrap_or(0),
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use http_client::{FakeHttpClient, Response};
    use std::sync::{Arc, Mutex};
    use time::Month;

    const USAGE_BODY: &str = r#"{
        "data": {
            "data": [
                {
                    "model": "z-ai/glm-5.3-flash-20260826",
                    "provider": "Together",
                    "cached_tokens": "2449174136",
                    "tokens_prompt": "2449174136",
                    "tokens_completion": "16840404",
                    "total_usage": 12.34
                },
                {
                    "model": "",
                    "provider": "",
                    "cached_tokens": "0",
                    "tokens_prompt": "0",
                    "tokens_completion": "0",
                    "total_usage": null
                },
                {
                    "model": "deepseek/deepseek-v4.1-flash-20261017",
                    "provider": "Wafer",
                    "tokens_prompt": "48766782",
                    "tokens_completion": "221651",
                    "total_usage": 5.67
                }
            ]
        }
    }"#;

    fn date(year: i32, month: u8, day: u8) -> OffsetDateTime {
        time::Date::from_calendar_date(year, Month::try_from(month).expect("month"), day)
            .expect("date")
            .midnight()
            .assume_utc()
    }

    async fn fetch(
        client: &dyn HttpClient,
        start: OffsetDateTime,
        end: OffsetDateTime,
    ) -> Result<Vec<AnalyticsRow>, OpenRouterError> {
        query_analytics(
            client,
            "secret",
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
        .await
    }

    #[test]
    fn query_analytics_parses_rows() {
        let client = FakeHttpClient::create(|_| async move {
            Ok(Response::builder()
                .status(200)
                .body(AsyncBody::from(USAGE_BODY))
                .expect("valid body"))
        });

        let rows = block_on(fetch(client.as_ref(), date(2026, 10, 2), date(2026, 10, 3)))
            .expect("analytics rows");

        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].start, date(2026, 10, 2));
        assert_eq!(rows[0].end, date(2026, 10, 3));
        assert_eq!(
            rows[0].model.as_deref(),
            Some("z-ai/glm-5.3-flash-20260826")
        );
        assert_eq!(rows[0].provider.as_deref(), Some("Together"));
        assert_eq!(rows[0].cached_tokens, Some(2_449_174_136));
        assert_eq!(rows[0].total_usage, Some(12.34));
        // A row without cached tokens reports none.
        assert_eq!(rows[2].provider.as_deref(), Some("Wafer"));
        assert_eq!(rows[2].cached_tokens, None);
    }

    #[test]
    fn query_analytics_splits_long_ranges() {
        let bodies = Arc::new(Mutex::new(Vec::new()));
        let captured = bodies.clone();
        let client = FakeHttpClient::create(move |request| {
            let captured = captured.clone();
            async move {
                let mut body = String::new();
                request
                    .into_body()
                    .read_to_string(&mut body)
                    .await
                    .expect("request body");
                captured.lock().expect("captured bodies lock").push(body);
                Ok(Response::builder()
                    .status(200)
                    .body(AsyncBody::from(USAGE_BODY))
                    .expect("valid body"))
            }
        });

        let rows = block_on(fetch(client.as_ref(), date(2026, 9, 1), date(2026, 10, 15)))
            .expect("analytics rows");

        // Every chunk reports the fixture rows, stamped with its window.
        assert_eq!(rows.len(), 6);
        assert_eq!(rows[0].start, date(2026, 9, 1));
        assert_eq!(rows[0].end, date(2026, 10, 2));
        assert_eq!(rows[3].start, date(2026, 10, 2));
        assert_eq!(rows[3].end, date(2026, 10, 15));
        let bodies = bodies.lock().expect("captured bodies lock");
        assert!(
            bodies[0].contains("2026-09-01T00:00:00Z")
                && bodies[0].contains("2026-10-02T00:00:00Z"),
            "first request body: {}",
            bodies[0]
        );
        assert!(
            bodies[1].contains("2026-10-02T00:00:00Z")
                && bodies[1].contains("2026-10-15T00:00:00Z"),
            "second request body: {}",
            bodies[1]
        );
    }

    #[test]
    fn query_analytics_surfaces_api_errors() {
        let client = FakeHttpClient::create(|_| async move {
            Ok(Response::builder().status(403).body(AsyncBody::from(
                r#"{"error":{"code":403,"message":"Only management keys can perform this operation"}}"#,
            ))?)
        });

        let error = block_on(fetch(client.as_ref(), date(2026, 10, 2), date(2026, 10, 3)))
            .expect_err("api error");

        let OpenRouterError::ApiError(api_error) = error else {
            panic!("expected ApiError, got {error:?}");
        };
        assert_eq!(api_error.status, Some(403));
        assert_eq!(
            api_error.message,
            "Only management keys can perform this operation"
        );
    }
}
