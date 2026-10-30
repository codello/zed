use futures::AsyncReadExt;
use http_client::{
    AsyncBody, CustomHeaders, HttpClient, Method, Request as HttpRequest, RequestBuilderExt,
};
use serde::{Deserialize, Deserializer};

use crate::{
    ApiError, OPEN_ROUTER_APP_TITLE, OpenRouterError, OpenRouterErrorBody, OpenRouterErrorResponse,
    retry_after_with_rate_limit_default,
};

/// A provider endpoint that serves a model.
#[derive(Clone, Debug, PartialEq)]
pub struct ProviderEndpoint {
    pub name: String,          // e.g. 'OpenAI: GPT-4'
    pub provider_name: String, // e.g. 'OpenAI'
    pub slug: String,          // e.g. 'openai'
    pub context_length: u64,
    pub max_completion_tokens: Option<u64>,
    pub quantization: Option<String>,
    pub service_tier: Option<String>,
    pub status: Option<i64>,
    pub supported_parameters: Vec<String>,
    pub pricing: ProviderPricing,
    pub throughput_last_30m: Option<PercentileStats>,
    pub latency_last_30m: Option<PercentileStats>,
}

/// Pricing for a provider endpoint, in USD per token.
#[derive(Clone, Debug, PartialEq)]
pub struct ProviderPricing {
    pub prompt: f64,
    pub completion: f64,
    pub input_cache_read: Option<f64>,
    /// The price of writing tokens to the cache, when the provider charges
    /// for writes.
    pub input_cache_write: Option<f64>,
    /// The fraction taken off the list price, between 0 and 1.
    pub discount: f64,
}

/// Percentiles of a metric over a time window.
///
/// For throughput, `pN` is the speed that `N` percent of requests reached or
/// exceeded, so `p90` is lower than `p50`. For latency, higher percentiles are
/// slower.
#[derive(Clone, Debug, PartialEq)]
pub struct PercentileStats {
    pub p50: f64,
    pub p75: f64,
    pub p90: f64,
    pub p99: f64,
}

#[derive(Debug, Deserialize)]
struct ModelEndpointsResponse {
    data: ModelEndpointsData,
}

#[derive(Debug, Deserialize)]
struct ModelEndpointsData {
    #[serde(default)]
    endpoints: Vec<ModelEndpointEntry>,
}

#[derive(Debug, Deserialize)]
struct ModelEndpointEntry {
    name: String,
    provider_name: String,
    tag: String,
    context_length: u64,
    #[serde(default)]
    max_completion_tokens: Option<u64>,
    #[serde(default)]
    quantization: Option<String>,
    #[serde(default)]
    service_tier: Option<String>,
    #[serde(default)]
    status: Option<i64>,
    supported_parameters: Vec<String>,
    pricing: ModelEndpointPricing,
    #[serde(default)]
    throughput_last_30m: Option<ModelEndpointPercentileStats>,
    #[serde(default)]
    latency_last_30m: Option<ModelEndpointPercentileStats>,
}

#[derive(Debug, Deserialize)]
struct ModelEndpointPercentileStats {
    #[serde(default)]
    p50: Option<f64>,
    #[serde(default)]
    p75: Option<f64>,
    #[serde(default)]
    p90: Option<f64>,
    #[serde(default)]
    p99: Option<f64>,
}

impl ModelEndpointPercentileStats {
    /// Returns `None` unless every percentile is present, so a partial payload
    /// hides the metric instead of failing the whole endpoint list.
    fn into_stats(self) -> Option<PercentileStats> {
        Some(PercentileStats {
            p50: self.p50?,
            p75: self.p75?,
            p90: self.p90?,
            p99: self.p99?,
        })
    }
}

#[derive(Debug, Deserialize)]
struct ModelEndpointPricing {
    #[serde(deserialize_with = "deserialize_price")]
    prompt: f64,
    #[serde(deserialize_with = "deserialize_price")]
    completion: f64,
    #[serde(default)]
    input_cache_read: Option<String>,
    #[serde(default)]
    input_cache_write: Option<String>,
    discount: Option<f64>,
}

/// OpenRouter reports prices as decimal strings.
fn deserialize_price<'de, D>(deserializer: D) -> Result<f64, D::Error>
where
    D: Deserializer<'de>,
{
    let value = String::deserialize(deserializer)?;
    value.parse().map_err(serde::de::Error::custom)
}

pub(crate) fn parse_price(value: Option<String>) -> Option<f64> {
    value.and_then(|value| value.parse().ok())
}

/// Lists the provider endpoints that serve `model_id`, for example
/// `openai/gpt-4`.
pub async fn list_provider_endpoints(
    client: &dyn HttpClient,
    api_url: &str,
    api_key: &str,
    model_id: &str,
    extra_headers: &CustomHeaders,
) -> Result<Vec<ProviderEndpoint>, OpenRouterError> {
    let uri = format!("{api_url}/models/{model_id}/endpoints");
    let request = HttpRequest::builder()
        .method(Method::GET)
        .uri(uri)
        .header("Accept", "application/json")
        .header("Authorization", format!("Bearer {}", api_key))
        .header("HTTP-Referer", "https://zed.dev")
        .header("X-Title", OPEN_ROUTER_APP_TITLE)
        .extra_headers(extra_headers)
        .body(AsyncBody::default())
        .map_err(OpenRouterError::BuildRequestBody)?;
    let host = request.uri().host().unwrap_or(api_url).to_owned();
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
        let response: ModelEndpointsResponse =
            serde_json::from_str(&body).map_err(OpenRouterError::DeserializeResponse)?;

        Ok(response
            .data
            .endpoints
            .into_iter()
            .map(|endpoint| ProviderEndpoint {
                name: endpoint.name,
                provider_name: endpoint.provider_name,
                slug: endpoint.tag,
                context_length: endpoint.context_length,
                max_completion_tokens: endpoint.max_completion_tokens,
                quantization: endpoint.quantization,
                service_tier: endpoint.service_tier,
                status: endpoint.status,
                supported_parameters: endpoint.supported_parameters,
                pricing: ProviderPricing {
                    prompt: endpoint.pricing.prompt,
                    completion: endpoint.pricing.completion,
                    input_cache_read: parse_price(endpoint.pricing.input_cache_read),
                    input_cache_write: parse_price(endpoint.pricing.input_cache_write),
                    discount: endpoint.pricing.discount.unwrap_or(0.0),
                },
                throughput_last_30m: endpoint
                    .throughput_last_30m
                    .and_then(ModelEndpointPercentileStats::into_stats),
                latency_last_30m: endpoint
                    .latency_last_30m
                    .and_then(ModelEndpointPercentileStats::into_stats),
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

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use http_client::{FakeHttpClient, Response};
    use std::sync::{Arc, Mutex};

    #[test]
    fn list_provider_endpoints_parses_endpoints() {
        let captured_uri = Arc::new(Mutex::new(None));
        let captured_uri_for_handler = captured_uri.clone();
        let client = FakeHttpClient::create(move |request| {
            let captured_uri = captured_uri_for_handler.clone();
            async move {
                captured_uri
                    .lock()
                    .expect("captured uri lock")
                    .replace(request.uri().to_string());
                Ok(Response::builder().status(200).body(AsyncBody::from(
                    r#"{
                        "data": {
                            "id": "openai/gpt-4",
                            "name": "GPT-4",
                            "endpoints": [
                                {
                                    "name": "OpenAI: GPT-4",
                                    "provider_name": "OpenAI",
                                    "tag": "openai",
                                    "context_length": 8192,
                                    "max_completion_tokens": 4096,
                                    "quantization": "fp16",
                                    "service_tier": "flex",
                                    "status": 0,
                                    "supported_parameters": ["tools", "temperature"],
                                    "pricing": {
                                        "prompt": "0.00003",
                                        "completion": "0.00006",
                                        "input_cache_read": "0.000015",
                                        "input_cache_write": "0.0000375",
                                        "discount": 0.25
                                    },
                                    "throughput_last_30m": {
                                        "p50": 68.4,
                                        "p75": 55.1,
                                        "p90": 41.2,
                                        "p99": 22.9
                                    },
                                    "latency_last_30m": {
                                        "p50": 460.0,
                                        "p75": 720.0,
                                        "p90": 1100.0,
                                        "p99": 2400.0
                                    }
                                },
                                {
                                    "name": "Azure: GPT-4",
                                    "provider_name": "Azure",
                                    "tag": "azure",
                                    "context_length": 8192,
                                    "max_completion_tokens": 4096,
                                    "quantization": null,
                                    "status": 0,
                                    "supported_parameters": ["tools"],
                                    "pricing": {
                                        "prompt": "0.00003",
                                        "completion": "0.00006"
                                    }
                                }
                            ]
                        }
                    }"#,
                ))?)
            }
        });

        let endpoints = block_on(list_provider_endpoints(
            client.as_ref(),
            crate::OPEN_ROUTER_API_URL,
            "secret",
            "openai/gpt-4",
            &CustomHeaders::default(),
        ))
        .expect("provider endpoints");

        assert_eq!(endpoints.len(), 2);
        assert_eq!(endpoints[0].provider_name, "OpenAI");
        assert_eq!(endpoints[0].slug, "openai");
        assert_eq!(endpoints[0].context_length, 8192);
        assert_eq!(endpoints[0].quantization.as_deref(), Some("fp16"));
        assert_eq!(endpoints[0].service_tier.as_deref(), Some("flex"));
        assert_eq!(
            endpoints[0].pricing,
            ProviderPricing {
                prompt: 0.00003,
                completion: 0.00006,
                input_cache_read: Some(0.000015),
                input_cache_write: Some(0.0000375),
                discount: 0.25,
            }
        );
        assert_eq!(endpoints[1].provider_name, "Azure");
        assert_eq!(endpoints[1].slug, "azure");
        assert_eq!(endpoints[1].quantization, None);
        assert_eq!(endpoints[1].service_tier, None);
        // A missing discount stands for no discount.
        assert_eq!(endpoints[1].pricing.discount, 0.0);
        assert_eq!(
            endpoints[0].throughput_last_30m,
            Some(PercentileStats {
                p50: 68.4,
                p75: 55.1,
                p90: 41.2,
                p99: 22.9,
            })
        );
        assert_eq!(
            endpoints[0].latency_last_30m,
            Some(PercentileStats {
                p50: 460.0,
                p75: 720.0,
                p90: 1100.0,
                p99: 2400.0,
            })
        );
        assert_eq!(endpoints[1].throughput_last_30m, None);
        assert_eq!(endpoints[1].latency_last_30m, None);
        assert_eq!(
            captured_uri.lock().expect("captured uri lock").as_deref(),
            Some("https://openrouter.ai/api/v1/models/openai/gpt-4/endpoints")
        );
    }

    #[test]
    fn list_provider_endpoints_surfaces_api_errors() {
        let client = FakeHttpClient::create(|_| async move {
            Ok(Response::builder().status(404).body(AsyncBody::from(
                r#"{"error":{"code":404,"message":"Resource not found"}}"#,
            ))?)
        });

        let error = block_on(list_provider_endpoints(
            client.as_ref(),
            crate::OPEN_ROUTER_API_URL,
            "secret",
            "openai/gpt-4",
            &CustomHeaders::default(),
        ))
        .expect_err("api error");

        let OpenRouterError::ApiError(api_error) = error else {
            panic!("expected ApiError, got {error:?}");
        };
        assert_eq!(api_error.status, Some(404));
        assert_eq!(api_error.code, 404);
        assert_eq!(api_error.message, "Resource not found");
    }
}
