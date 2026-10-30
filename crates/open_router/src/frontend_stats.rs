use std::collections::HashMap;

use futures::AsyncReadExt;
use http_client::{AsyncBody, HttpClient, Method, Request as HttpRequest, Url};
use language_model_core::{DEFAULT_OUTPUT_INPUT_RATIO, UpstreamProviderPricing};
use serde::Deserialize;

use crate::{
    ApiError, OpenRouterError, OpenRouterErrorBody, OpenRouterErrorResponse,
    provider_endpoints::parse_price, retry_after_with_rate_limit_default,
};

/// Base URL of OpenRouter's website API. Unlike the versioned API it is
/// unauthenticated, so requests must not carry the API key, and it is not
/// covered by the `api_url` setting.
const FRONTEND_STATS_URL: &str = "https://openrouter.ai/api/frontend/v1";

/// The origin that relative icon URLs resolve against.
const OPENROUTER_ORIGIN: &str = "https://openrouter.ai";

/// An endpoint that serves a model, as reported by OpenRouter's website API.
#[derive(Clone, Debug, PartialEq)]
pub struct FrontendEndpoint {
    /// The name of the provider operating the endpoint as OpenRouter reports
    /// it, for example `BaseTen`. This is the name that the analytics API
    /// aggregates on.
    pub provider_name: Option<String>,
    /// The average cache hit rate of the last week, as a fraction between 0
    /// and 1. `None` when the endpoint has no data.
    pub cache_hit_rate: Option<f64>,
    /// The region the endpoint runs in, for example `us` or `eu`, when
    /// reported.
    pub region: Option<String>,
    /// The service tier OpenRouter derived for the endpoint, for example
    /// `spot`.
    pub service_tier: Option<String>,
    /// The country the company operating the endpoint is headquartered in,
    /// for example `us`, when reported.
    pub headquarters: Option<String>,
    /// The countries the endpoint processes requests in.
    pub datacenters: Vec<String>,
    /// URL of the provider's icon, when reported. Relative values resolve
    /// against the OpenRouter origin.
    pub icon_url: Option<String>,
    /// The prices of the endpoint, in USD per token, when reported.
    pub pricing: Option<FrontendPricing>,
}

/// The prices of an endpoint, in USD per token.
#[derive(Clone, Debug, PartialEq)]
pub struct FrontendPricing {
    pub prompt: f64,
    pub completion: f64,
    /// The price of reading tokens from the cache, when the provider charges
    /// for reads.
    pub input_cache_read: Option<f64>,
    /// The price of writing tokens to the cache, when the provider charges
    /// for writes.
    pub input_cache_write: Option<f64>,
    /// The fraction taken off the list price, between 0 and 1. Zero when
    /// the endpoint reports none.
    pub discount: f64,
}

impl From<&FrontendPricing> for UpstreamProviderPricing {
    /// The pricing in USD per million tokens, with the default
    /// output-to-input ratio.
    fn from(pricing: &FrontendPricing) -> Self {
        UpstreamProviderPricing {
            input: pricing.prompt * 1e6,
            output: pricing.completion * 1e6,
            cache_read: pricing.input_cache_read.map(|price| price * 1e6),
            cache_write: pricing.input_cache_write.map(|price| price * 1e6),
            discount: pricing.discount,
            cache_hit_rate: None,
            output_input_ratio: DEFAULT_OUTPUT_INPUT_RATIO,
        }
    }
}

impl From<UpstreamProviderPricing> for FrontendPricing {
    /// The pricing in USD per token. The cache hit rate and the blended
    /// price have no frontend counterpart.
    fn from(pricing: UpstreamProviderPricing) -> Self {
        FrontendPricing {
            prompt: pricing.input / 1e6,
            completion: pricing.output / 1e6,
            input_cache_read: pricing.cache_read.map(|price| price / 1e6),
            input_cache_write: pricing.cache_write.map(|price| price / 1e6),
            discount: pricing.discount,
        }
    }
}

#[derive(Debug, Deserialize)]
struct FrontendEndpointsResponse {
    #[serde(default)]
    data: Vec<FrontendEndpointEntry>,
}

#[derive(Debug, Deserialize, PartialEq)]
struct FrontendEndpointEntry {
    #[serde(default)]
    id: String,
    #[serde(default)]
    provider_slug: String,
    #[serde(default)]
    provider_name: Option<String>,
    #[serde(default)]
    provider_region: Option<String>,
    #[serde(default)]
    derived_service_tier: Option<String>,
    #[serde(default)]
    pricing: Option<FrontendPricingEntry>,
    #[serde(default)]
    provider_info: Option<ProviderInfoEntry>,
}

#[derive(Debug, Deserialize, PartialEq)]
struct FrontendPricingEntry {
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default)]
    completion: Option<String>,
    #[serde(default)]
    input_cache_read: Option<String>,
    #[serde(default)]
    input_cache_write: Option<String>,
    #[serde(default)]
    discount: Option<f64>,
}

#[derive(Debug, Deserialize, PartialEq)]
struct ProviderInfoEntry {
    #[serde(default)]
    headquarters: Option<String>,
    #[serde(default)]
    datacenters: Vec<String>,
    #[serde(default)]
    icon: Option<ProviderIconEntry>,
}

#[derive(Debug, Deserialize, PartialEq)]
struct ProviderIconEntry {
    #[serde(default)]
    url: Option<String>,
}

#[derive(Debug, Deserialize)]
struct CacheHitRatesResponse {
    #[serde(default)]
    data: Vec<CacheHitRatePoint>,
}

#[derive(Debug, Deserialize)]
struct CacheHitRatePoint {
    #[serde(default)]
    y: HashMap<String, f64>,
}

/// Returns the endpoints that serve `permaslug` according to OpenRouter's
/// website API, keyed by the provider slug with its variant (for example
/// `relace/fp4`), which matches the `tag` of the versioned endpoints API.
/// Endpoints that the website API does not know are absent.
///
/// The permaslug is the model's canonical slug, for example
/// `deepseek/deepseek-v4.1-flash-20261017`; model ids from the versioned API
/// are aliases that this API does not resolve.
pub async fn list_frontend_endpoints(
    client: &dyn HttpClient,
    permaslug: &str,
) -> Result<HashMap<String, FrontendEndpoint>, OpenRouterError> {
    let (entries, rates) = futures::future::try_join(
        list_frontend_endpoint_entries(client, permaslug),
        list_endpoint_cache_hit_rates(client, permaslug),
    )
    .await?;
    // The website API keys its stats on opaque endpoint ids; the provider
    // slug is the stable handle that the versioned API shares.
    let mut endpoints = HashMap::new();
    for entry in entries {
        endpoints.insert(
            entry.provider_slug,
            FrontendEndpoint {
                provider_name: entry.provider_name,
                cache_hit_rate: rates.get(&entry.id).copied(),
                region: entry.provider_region,
                service_tier: entry.derived_service_tier,
                // Endpoints without input and output prices get no pricing.
                pricing: entry.pricing.and_then(|pricing| {
                    Some(FrontendPricing {
                        prompt: parse_price(pricing.prompt)?,
                        completion: parse_price(pricing.completion)?,
                        input_cache_read: parse_price(pricing.input_cache_read),
                        input_cache_write: parse_price(pricing.input_cache_write),
                        discount: pricing.discount.unwrap_or(0.0),
                    })
                }),
                headquarters: entry
                    .provider_info
                    .as_ref()
                    .and_then(|info| info.headquarters.clone()),
                datacenters: entry
                    .provider_info
                    .as_ref()
                    .map(|info| info.datacenters.clone())
                    .unwrap_or_default(),
                icon_url: entry
                    .provider_info
                    .and_then(|info| info.icon.and_then(|icon| icon.url))
                    .filter(|url| !url.is_empty())
                    .and_then(|url| resolve_icon_url(&url)),
            },
        );
    }
    Ok(endpoints)
}

/// Lists the raw entries of the endpoints that serve `permaslug` according
/// to OpenRouter's website API.
async fn list_frontend_endpoint_entries(
    client: &dyn HttpClient,
    permaslug: &str,
) -> Result<Vec<FrontendEndpointEntry>, OpenRouterError> {
    let body = get_frontend_stats(
        client,
        "stats/endpoint",
        &[
            ("latencyMetric", "latency"),
            ("perfWorkload", "text_generation"),
            ("permaslug", permaslug),
            ("variant", "standard"),
        ],
    )
    .await?;
    let response: FrontendEndpointsResponse =
        serde_json::from_str(&body).map_err(OpenRouterError::DeserializeResponse)?;

    Ok(response.data)
}

/// Returns the average cache hit rate of the last week for each endpoint id,
/// as a fraction between 0 and 1. Endpoints without data are absent.
async fn list_endpoint_cache_hit_rates(
    client: &dyn HttpClient,
    permaslug: &str,
) -> Result<HashMap<String, f64>, OpenRouterError> {
    let body = get_frontend_stats(
        client,
        "stats/cache-hit-rate-comparison",
        &[("permaslug", permaslug), ("timeRange", "1w")],
    )
    .await?;
    let response: CacheHitRatesResponse =
        serde_json::from_str(&body).map_err(OpenRouterError::DeserializeResponse)?;

    // The series values are percentages. Days without an entry had no
    // traffic and do not count toward the average.
    let mut totals: HashMap<String, (f64, u32)> = HashMap::new();
    for point in response.data {
        for (id, rate) in point.y {
            let entry = totals.entry(id).or_insert((0.0, 0));
            entry.0 += rate;
            entry.1 += 1;
        }
    }
    Ok(totals
        .into_iter()
        .map(|(id, (total, days))| (id, total / f64::from(days) / 100.0))
        .collect())
}

/// Resolves an icon URL against the OpenRouter origin. Absolute URLs are
/// kept as they are; values that cannot be resolved are dropped.
fn resolve_icon_url(icon_url: &str) -> Option<String> {
    Url::parse(OPENROUTER_ORIGIN)
        .ok()?
        .join(icon_url)
        .ok()
        .map(|url| url.to_string())
}

/// Fetches `path` from OpenRouter's website API with `query` appended as
/// percent-encoded query parameters.
async fn get_frontend_stats(
    client: &dyn HttpClient,
    path: &str,
    query: &[(&str, &str)],
) -> Result<String, OpenRouterError> {
    let url = Url::parse_with_params(
        &format!("{FRONTEND_STATS_URL}/{path}"),
        query.iter().copied(),
    )
    .map_err(OpenRouterError::BuildUrl)?;
    let request = HttpRequest::builder()
        .method(Method::GET)
        .uri(url.as_str())
        .header("Accept", "application/json")
        .body(AsyncBody::default())
        .map_err(OpenRouterError::BuildRequestBody)?;
    let host = request.uri().host().unwrap_or("openrouter.ai").to_owned();
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
        Ok(body)
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

    const FRONTEND_ENDPOINTS_BODY: &str = r#"{
        "data": [
            {
                "id": "913e3d9a-dc7e-4e84-b892-35169efa3de3",
                "name": "Wafer | deepseek/deepseek-v4.1-flash-20261017",
                "provider_slug": "wafer/us",
                "provider_name": "Wafer",
                "provider_region": "us",
                "quantization": "mxfp4",
                "pricing": {
                    "prompt": "0.00000004",
                    "completion": "0.0000002",
                    "input_cache_read": "0.000000004",
                    "discount": 0.25
                },
                "provider_info": {
                    "headquarters": "US",
                    "datacenters": ["US"],
                    "icon": {"url": "https://example.com/wafer.png"}
                }
            },
            {
                "id": "e768a23f-2848-4af0-a7c0-c02cf79ed41c",
                "name": "Morph | deepseek/deepseek-v4.1-flash-20261017",
                "provider_slug": "morph/fp8",
                "provider_name": "Morph"
            },
            {
                "id": "no-rates-endpoint",
                "name": "NoRates | deepseek/deepseek-v4.1-flash-20261017",
                "provider_slug": "no-rates/fp8",
                "derived_service_tier": "spot",
                "provider_info": {
                    "icon": {"url": "/images/icons/NoRates.png"}
                }
            },
            {
                "id": "partial-pricing-endpoint",
                "name": "Partial | deepseek/deepseek-v4.1-flash-20261017",
                "provider_slug": "partial/fp8",
                "pricing": {"completion": "0.0000002"}
            }
        ]
    }"#;

    const CACHE_HIT_RATES_BODY: &str = r#"{
        "data": [
            {
                "x": "2026-10-25 00:00:00",
                "y": {
                    "913e3d9a-dc7e-4e84-b892-35169efa3de3": 90.0,
                    "e768a23f-2848-4af0-a7c0-c02cf79ed41c": 60.0
                }
            },
            {
                "x": "2026-10-26 00:00:00",
                "y": {"913e3d9a-dc7e-4e84-b892-35169efa3de3": 100.0}
            }
        ]
    }"#;

    #[test]
    fn list_frontend_endpoint_entries_parses_entries() {
        let captured = Arc::new(Mutex::new(None));
        let captured_for_handler = captured.clone();
        let client = FakeHttpClient::create(move |request| {
            let captured = captured_for_handler.clone();
            async move {
                captured
                    .lock()
                    .expect("captured request lock")
                    .replace(request.uri().to_string());
                Ok(Response::builder()
                    .status(200)
                    .body(AsyncBody::from(FRONTEND_ENDPOINTS_BODY))
                    .expect("valid body"))
            }
        });

        let endpoints = block_on(list_frontend_endpoint_entries(
            client.as_ref(),
            "deepseek/deepseek-v4.1-flash-20261017",
        ))
        .expect("frontend endpoints");

        assert_eq!(
            endpoints,
            vec![
                FrontendEndpointEntry {
                    id: "913e3d9a-dc7e-4e84-b892-35169efa3de3".to_string(),
                    provider_slug: "wafer/us".to_string(),
                    provider_name: Some("Wafer".to_string()),
                    provider_region: Some("us".to_string()),
                    derived_service_tier: None,
                    pricing: Some(FrontendPricingEntry {
                        prompt: Some("0.00000004".to_string()),
                        completion: Some("0.0000002".to_string()),
                        input_cache_read: Some("0.000000004".to_string()),
                        input_cache_write: None,
                        discount: Some(0.25),
                    }),
                    provider_info: Some(ProviderInfoEntry {
                        headquarters: Some("US".to_string()),
                        datacenters: vec!["US".to_string()],
                        icon: Some(ProviderIconEntry {
                            url: Some("https://example.com/wafer.png".to_string()),
                        }),
                    }),
                },
                FrontendEndpointEntry {
                    id: "e768a23f-2848-4af0-a7c0-c02cf79ed41c".to_string(),
                    provider_slug: "morph/fp8".to_string(),
                    provider_name: Some("Morph".to_string()),
                    provider_region: None,
                    derived_service_tier: None,
                    pricing: None,
                    provider_info: None,
                },
                FrontendEndpointEntry {
                    id: "no-rates-endpoint".to_string(),
                    provider_slug: "no-rates/fp8".to_string(),
                    provider_name: None,
                    provider_region: None,
                    derived_service_tier: Some("spot".to_string()),
                    pricing: None,
                    provider_info: Some(ProviderInfoEntry {
                        headquarters: None,
                        datacenters: Vec::new(),
                        icon: Some(ProviderIconEntry {
                            url: Some("/images/icons/NoRates.png".to_string()),
                        }),
                    }),
                },
                FrontendEndpointEntry {
                    id: "partial-pricing-endpoint".to_string(),
                    provider_slug: "partial/fp8".to_string(),
                    provider_name: None,
                    provider_region: None,
                    derived_service_tier: None,
                    pricing: Some(FrontendPricingEntry {
                        prompt: None,
                        completion: Some("0.0000002".to_string()),
                        input_cache_read: None,
                        input_cache_write: None,
                        discount: None,
                    }),
                    provider_info: None,
                },
            ]
        );
        let uri = captured
            .lock()
            .expect("captured request lock")
            .clone()
            .expect("captured request");
        assert!(uri.contains("permaslug=deepseek%2Fdeepseek-v4.1-flash-20261017"));
        assert!(!uri.contains("Authorization"));
    }

    #[test]
    fn list_endpoint_cache_hit_rates_averages_the_week() {
        let client = FakeHttpClient::create(|request| {
            let uri = request.uri().to_string();
            async move {
                assert!(uri.contains("cache-hit-rate-comparison"));
                assert!(uri.contains("timeRange=1w"));
                assert!(uri.contains("permaslug=z-ai%2Fglm-5.3-flash-20260826"));
                Ok(Response::builder()
                    .status(200)
                    .body(AsyncBody::from(CACHE_HIT_RATES_BODY))
                    .expect("valid body"))
            }
        });

        let rates = block_on(list_endpoint_cache_hit_rates(
            client.as_ref(),
            "z-ai/glm-5.3-flash-20260826",
        ))
        .expect("cache hit rates");

        assert_eq!(rates.len(), 2);
        assert!((rates["913e3d9a-dc7e-4e84-b892-35169efa3de3"] - 0.95).abs() < 1e-9);
        assert!((rates["e768a23f-2848-4af0-a7c0-c02cf79ed41c"] - 0.6).abs() < 1e-9);
    }

    #[test]
    fn list_frontend_endpoints_join_rates_by_provider_slug() {
        let client = FakeHttpClient::create(|request| {
            let path = request.uri().path().to_string();
            async move {
                if path.contains("/stats/endpoint") {
                    Ok(Response::builder()
                        .status(200)
                        .body(AsyncBody::from(FRONTEND_ENDPOINTS_BODY))
                        .expect("valid body"))
                } else {
                    Ok(Response::builder()
                        .status(200)
                        .body(AsyncBody::from(CACHE_HIT_RATES_BODY))
                        .expect("valid body"))
                }
            }
        });

        let endpoints = block_on(list_frontend_endpoints(
            client.as_ref(),
            "deepseek/deepseek-v4.1-flash-20261017",
        ))
        .expect("frontend endpoints");

        // The rates are keyed by the provider slug, not by the endpoint id.
        assert_eq!(endpoints.len(), 4);
        assert!((endpoints["wafer/us"].cache_hit_rate.unwrap() - 0.95).abs() < 1e-9);
        assert_eq!(endpoints["wafer/us"].region.as_deref(), Some("us"));
        assert_eq!(
            endpoints["wafer/us"].provider_name.as_deref(),
            Some("Wafer")
        );
        assert_eq!(endpoints["wafer/us"].headquarters.as_deref(), Some("US"));
        assert_eq!(endpoints["wafer/us"].datacenters, vec!["US".to_string()]);
        assert_eq!(
            endpoints["wafer/us"].icon_url.as_deref(),
            Some("https://example.com/wafer.png")
        );
        let pricing = endpoints["wafer/us"]
            .pricing
            .as_ref()
            .expect("the wafer endpoint has prices");
        assert_eq!(pricing.prompt, 0.00000004);
        assert_eq!(pricing.completion, 0.0000002);
        assert_eq!(pricing.input_cache_read, Some(0.000000004));
        assert_eq!(pricing.input_cache_write, None);
        assert_eq!(pricing.discount, 0.25);
        // An endpoint without input and output prices gets no pricing.
        assert_eq!(endpoints["partial/fp8"].pricing, None);
        assert_eq!(endpoints["morph/fp8"].headquarters, None);
        assert!((endpoints["morph/fp8"].cache_hit_rate.unwrap() - 0.6).abs() < 1e-9);
        assert_eq!(endpoints["no-rates/fp8"].cache_hit_rate, None);
        assert_eq!(
            endpoints["no-rates/fp8"].service_tier.as_deref(),
            Some("spot")
        );
        // A relative icon URL resolves against the OpenRouter origin.
        assert_eq!(
            endpoints["no-rates/fp8"].icon_url.as_deref(),
            Some("https://openrouter.ai/images/icons/NoRates.png")
        );
    }

    #[test]
    fn frontend_pricing_converts_to_upstream_pricing() {
        let pricing = FrontendPricing {
            prompt: 0.00003,
            completion: 0.00006,
            input_cache_read: Some(0.000015),
            input_cache_write: None,
            discount: 0.25,
        };

        let upstream = UpstreamProviderPricing::from(&pricing);
        assert_eq!(upstream.input, 30.0);
        assert_eq!(upstream.output, 60.0);
        assert_eq!(upstream.cache_read, Some(15.0));
        assert_eq!(upstream.cache_write, None);
        assert_eq!(upstream.discount, 0.25);
        assert_eq!(upstream.cache_hit_rate, None);
        assert_eq!(upstream.output_input_ratio, DEFAULT_OUTPUT_INPUT_RATIO);
        // The reverse conversion reproduces the per-token prices.
        assert_eq!(FrontendPricing::from(upstream), pricing);
    }

    #[test]
    fn frontend_stats_requests_carry_no_api_key() {
        let client = FakeHttpClient::create(|request| {
            assert!(request.headers().get("Authorization").is_none());
            async move {
                Ok(Response::builder()
                    .status(200)
                    .body(AsyncBody::from(FRONTEND_ENDPOINTS_BODY))
                    .expect("valid body"))
            }
        });

        block_on(list_frontend_endpoints(client.as_ref(), "openai/gpt-4"))
            .expect("frontend endpoints");
    }

    #[test]
    fn frontend_stats_surfaces_api_errors() {
        let client = FakeHttpClient::create(|_| async move {
            Ok(Response::builder().status(404).body(AsyncBody::from(
                r#"{"error":{"code":404,"message":"Resource not found"}}"#,
            ))?)
        });

        let error = block_on(list_frontend_endpoints(client.as_ref(), "openai/gpt-4"))
            .expect_err("api error");

        let OpenRouterError::ApiError(api_error) = error else {
            panic!("expected ApiError, got {error:?}");
        };
        assert_eq!(api_error.status, Some(404));
        assert_eq!(api_error.message, "Resource not found");
    }
}
