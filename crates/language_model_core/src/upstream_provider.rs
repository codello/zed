//! Pricing and metadata for an upstream provider that can serve a model.

/// The fraction of input tokens written to the cache. We currently have no
/// source of data for this.
const CACHE_WRITE_FRACTION: f64 = 0.0;

/// The output-to-input token ratio assumed for models without measured
/// statistics. The upstream provider selector serves the agent panel, whose
/// workload is input-heavy.
pub const DEFAULT_OUTPUT_INPUT_RATIO: f64 = 0.01;

/// An upstream provider that can serve a model.
#[derive(Clone, Debug, PartialEq)]
pub struct UpstreamProvider {
    /// The value used in provider routing, for example `openai`.
    pub id: String,
    /// The name shown to users, for example `OpenAI`.
    pub name: String,
    /// The service tier of the endpoint, for example `flex` or `spot`, when
    /// it is not the standard tier.
    pub service_tier: Option<String>,
    /// The quantization the endpoint runs with, for example `fp8`, when
    /// known.
    pub quantization: Option<String>,
    /// The region the endpoint runs in, for example `us` or `eu`, when
    /// reported.
    pub region: Option<String>,
    /// The country the company operating the endpoint is headquartered in,
    /// for example `us`, when reported.
    pub headquarters: Option<String>,
    /// The countries the endpoint processes requests in.
    pub datacenters: Vec<String>,
    /// URL of the provider's icon, when reported.
    pub icon_url: Option<String>,
    pub pricing: UpstreamProviderPricing,
    pub throughput: Option<f64>,
    pub latency: Option<f64>,
    pub context_length: u64,
}

/// The prices of an upstream provider, in USD per million tokens, together
/// with the workload assumptions used to estimate the expected cost.
#[derive(Clone, Debug, PartialEq)]
pub struct UpstreamProviderPricing {
    pub input: f64,
    pub output: f64,
    pub cache_read: Option<f64>,
    /// The price of writing tokens to the cache, when the provider charges
    /// for writes.
    pub cache_write: Option<f64>,
    /// The fraction taken off the list price.
    pub discount: f64,
    /// The cache hit rate used to compute [`Self::effective_input`]: the
    /// global rate of the last week, scaled by the workload calibration.
    pub cache_hit_rate: Option<f64>,
    /// The output-to-input token ratio of the workload, used to compute
    /// [`Self::blended`].
    pub output_input_ratio: f64,
}

impl UpstreamProviderPricing {
    /// The input price after weighting by the expected cache hit rate.
    ///
    /// An endpoint that cannot cache reads at the list input price; for an
    /// endpoint with unknown cache pricing the list input price stands in.
    pub fn effective_input(&self) -> f64 {
        match (self.cache_hit_rate, self.cache_read) {
            (Some(rate), Some(cache_read)) => {
                let uncached = 1.0 - rate - CACHE_WRITE_FRACTION;
                uncached * self.input
                    + rate * cache_read
                    + CACHE_WRITE_FRACTION * self.cache_write.unwrap_or(self.input)
            }
            _ => self.input,
        }
    }

    /// The estimated cost per million input tokens: the effective input
    /// price plus the output price weighted by the output-to-input token
    /// ratio.
    pub fn blended(&self) -> f64 {
        self.effective_input() + self.output_input_ratio * self.output
    }
}
