mod helpers;

use anyhow::Result;
use futures::{AsyncBufReadExt, AsyncReadExt, StreamExt, io::BufReader, stream::BoxStream};
use helpers::{extract_retry_after, is_none_or_empty};
use http_client::{AsyncBody, HttpClient, Method, Request as HttpRequest, http};
use language_model_core::ReasoningEffort;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{io, time::Duration};

pub const LITELLM_API_URL: &str = "https://aikeys.maibornwolff.de";

/// Identifies which upstream provider family a model belongs to, used to gate
/// provider-specific request fields such as Anthropic's `cache_control`.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Dialect {
    Anthropic,
    OpenAi,
    Google,
}

// ---------- Model listing types ----------

/// Deserializes a list, treating an explicit `null` as empty; the proxy sends
/// `null` for unset list fields on some models.
fn null_as_default<'de, D>(deserializer: D) -> Result<Vec<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Ok(Option::<Vec<String>>::deserialize(deserializer)?.unwrap_or_default())
}

#[derive(Default, Debug, Clone, Deserialize)]
pub struct ListModelsResponse {
    pub data: Vec<ModelInfo>,
}

/// Raw model info from LiteLLM's /model_group/info endpoint.
#[derive(Default, Debug, Clone, Deserialize)]
pub struct ModelInfo {
    #[serde(rename = "model_group")]
    pub name: String,
    pub mode: Option<String>,
    #[serde(default, deserialize_with = "null_as_default")]
    pub providers: Vec<String>,
    #[serde(default)]
    pub supports_vision: bool,
    #[serde(default, deserialize_with = "null_as_default")]
    pub supported_openai_params: Vec<String>,
    pub supported_reasoning_efforts: Option<Vec<String>>,
    pub max_input_tokens: Option<f64>,
    pub max_output_tokens: Option<f64>,
    pub input_cost_per_token: Option<f64>,
    pub output_cost_per_token: Option<f64>,
}

/// A model derived from LiteLLM's model listing API.
#[derive(Default, Debug, Clone, PartialEq)]
pub struct Model {
    pub name: String,
    pub display_name: String,
    pub max_input_tokens: u64,
    pub max_output_tokens: u64,
    pub supports_tools: bool,
    pub supports_parallel_tool_calls: bool,
    pub supports_tool_choice: bool,
    pub supports_vision: bool,
    pub supports_thinking: bool,
    pub supports_reasoning_effort: bool,
    pub supported_reasoning_efforts: Option<Vec<ReasoningEffort>>,
    pub supports_context_management: bool,
    pub input_cost_per_token: Option<f64>,
    pub output_cost_per_token: Option<f64>,
}

impl Model {
    pub fn display_name(&self) -> &str {
        if self.display_name.is_empty() {
            &self.name
        } else {
            &self.display_name
        }
    }

    pub fn max_token_count(&self) -> u64 {
        self.max_input_tokens
    }

    pub fn max_output_tokens(&self) -> Option<u64> {
        if self.max_output_tokens == 0 {
            None
        } else {
            Some(self.max_output_tokens)
        }
    }

    /// Returns which upstream provider family this model belongs to,
    /// or None if it's a custom/unknown model.
    pub fn dialect(&self) -> Option<Dialect> {
        if self.name.starts_with("claude-") {
            Some(Dialect::Anthropic)
        } else if self.name.starts_with("gpt-")
            || self.name.starts_with("o3")
            || self.name.starts_with("o4")
        {
            Some(Dialect::OpenAi)
        } else if self.name.starts_with("gemini-") {
            Some(Dialect::Google)
        } else {
            None
        }
    }

    /// The efforts this model accepts for the `reasoning_effort` parameter;
    /// empty when it doesn't support effort-based reasoning.
    pub fn reasoning_efforts(&self) -> &[ReasoningEffort] {
        match &self.supported_reasoning_efforts {
            Some(efforts) => efforts,
            None if self.supports_reasoning_effort => {
                &ReasoningEffort::OPENAI_COMPATIBLE_SELECTABLE
            }
            None => &[],
        }
    }

    /// The effort to use when the request doesn't select one: `high` when
    /// supported, otherwise the first supported effort.
    pub fn default_reasoning_effort(&self) -> Option<ReasoningEffort> {
        let efforts = self.reasoning_efforts();
        if efforts.contains(&ReasoningEffort::High) {
            Some(ReasoningEffort::High)
        } else {
            efforts.first().copied()
        }
    }
}

/// Fetches the list of available models from a LiteLLM proxy.
/// Uses the /model_group/info endpoint and authenticates with X-LiteLLM-Api-Key.
pub async fn fetch_models(
    client: &dyn HttpClient,
    api_url: &str,
    api_key: Option<&str>,
) -> Result<Vec<Model>, LiteLlmError> {
    let mut request_builder = HttpRequest::builder()
        .method(Method::GET)
        .uri(format!("{api_url}/model_group/info"))
        .header("Accept", "application/json");
    if let Some(api_key) = api_key {
        request_builder = request_builder.header("X-LiteLLM-Api-Key", api_key);
    }
    let request = request_builder
        .body(AsyncBody::empty())
        .map_err(LiteLlmError::BuildRequestBody)?;
    let host = request.uri().host().unwrap_or(api_url).to_owned();
    let mut response = client
        .send(request)
        .await
        .map_err(|error| LiteLlmError::HttpSend { host, error })?;

    let mut body = String::new();
    response
        .body_mut()
        .read_to_string(&mut body)
        .await
        .map_err(LiteLlmError::ReadResponse)?;

    if !response.status().is_success() {
        return Err(error_from_response(response, body));
    }

    let response: ListModelsResponse =
        serde_json::from_str(&body).map_err(LiteLlmError::DeserializeResponse)?;

    Ok(response
        .data
        .into_iter()
        .filter(|entry| {
            entry.mode == None
                || entry.mode == Some("chat".into())
                || entry.mode == Some("responses".into())
        })
        .map(|entry| {
            // The proxy sends `null` params for some models, which
            // deserializes to an empty list; treat that as "not reported" and
            // assume the common capability set rather than disabling
            // everything.
            let supports = |param: &str| {
                entry.supported_openai_params.is_empty()
                    || entry
                        .supported_openai_params
                        .iter()
                        .any(|candidate| candidate.as_str() == param)
            };
            Model {
                name: entry.name.clone(),
                display_name: entry.name,
                supports_tools: supports("tools"),
                supports_parallel_tool_calls: supports("parallel_tool_calls"),
                supports_tool_choice: supports("tool_choice"),
                supports_vision: entry.supports_vision,
                supports_thinking: supports("thinking"),
                supports_reasoning_effort: supports("reasoning_effort"),
                supported_reasoning_efforts: entry.supported_reasoning_efforts.map(|efforts| {
                    efforts
                        .into_iter()
                        .filter_map(|effort| effort.parse().ok())
                        .collect()
                }),
                supports_context_management: supports("context_management"),
                max_input_tokens: entry.max_input_tokens.unwrap_or(4_096.0) as u64,
                max_output_tokens: entry.max_output_tokens.unwrap_or(4_096.0) as u64,
                input_cost_per_token: entry.input_cost_per_token,
                output_cost_per_token: entry.output_cost_per_token,
            }
        })
        .collect())
}

// ---------- OpenAI-compatible request/response types ----------

#[derive(Clone, Copy, Serialize, Deserialize, Debug, Eq, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
    System,
    Tool,
}

/// An OpenAI-compatible chat completion request, with LiteLLM custom parameters.
#[derive(Debug, Serialize, Deserialize)]
pub struct Request {
    pub model: String,
    pub messages: Vec<RequestMessage>,
    pub stream: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stop: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ToolDefinition>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<Thinking>,
    /// Top-level automatic prompt caching. LiteLLM forwards this to Anthropic,
    /// which places a cache breakpoint on the last cacheable block automatically.
    /// Ignored by non-Anthropic backends.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
    /// Server-side context compaction settings. LiteLLM forwards this to supported
    /// backends (Anthropic, Azure AI, Vertex AI) and automatically adds the required
    /// beta header. Not supported on Bedrock.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_management: Option<ContextManagement>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ToolChoice {
    Auto,
    Required,
    None,
    #[serde(untagged)]
    Other(ToolDefinition),
}

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Deserialize, Serialize, Debug)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolDefinition {
    #[allow(dead_code)]
    Function {
        function: FunctionDefinition,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
}

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FunctionDefinition {
    pub name: String,
    pub description: Option<String>,
    pub parameters: Option<Value>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct StreamOptions {
    pub include_usage: bool,
}

impl Default for StreamOptions {
    fn default() -> Self {
        Self {
            include_usage: true,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Eq, PartialEq)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Thinking {
    Disabled,
    #[serde(rename = "enabled")]
    Manual {
        budget_tokens: u32,
    },
    Adaptive {
        display: Option<ThinkingDisplay>,
    },
}

#[derive(Serialize, Deserialize, Debug, Eq, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ThinkingDisplay {
    Omitted,
    Summarized,
}

#[derive(Serialize, Deserialize, Debug, Eq, PartialEq)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum RequestMessage {
    Assistant {
        content: Option<MessageContent>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        tool_calls: Vec<ToolCall>,
        /// Thinking blocks to include in the assistant message for Anthropic models.
        /// Required when using extended thinking with tool calls, so Anthropic can
        /// validate the signature on the next turn.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        thinking_blocks: Vec<ThinkingBlock>,
        /// Compaction blocks to replay in the assistant message. LiteLLM forwards these
        /// to Anthropic (and other supported backends) via `provider_specific_fields`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        provider_specific_fields: Option<ProviderSpecificFields>,
    },
    User {
        content: MessageContent,
    },
    System {
        content: MessageContent,
    },
    Tool {
        content: MessageContent,
        tool_call_id: String,
    },
}

#[derive(Serialize, Deserialize, Debug, Eq, PartialEq)]
#[serde(untagged)]
pub enum MessageContent {
    Plain(String),
    Multipart(Vec<MessagePart>),
}

impl MessageContent {
    pub fn empty() -> Self {
        Self::Plain(String::new())
    }

    pub fn push_part(&mut self, part: MessagePart) {
        match self {
            Self::Plain(text) if text.is_empty() => {
                *self = Self::Multipart(vec![part]);
            }
            Self::Plain(text) => {
                let text_part = MessagePart::Text {
                    text: std::mem::take(text),
                    cache_control: None,
                };
                *self = Self::Multipart(vec![text_part, part]);
            }
            Self::Multipart(parts) => parts.push(part),
        }
    }

    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Plain(text) => Some(text),
            Self::Multipart(parts) if parts.len() == 1 => {
                if let MessagePart::Text { text, .. } = &parts[0] {
                    Some(text)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    pub fn to_text(&self) -> String {
        match self {
            Self::Plain(text) => text.clone(),
            Self::Multipart(parts) => parts
                .iter()
                .filter_map(|part| {
                    if let MessagePart::Text { text, .. } = part {
                        Some(text.as_str())
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
                .join(""),
        }
    }
}

impl From<Vec<MessagePart>> for MessageContent {
    fn from(parts: Vec<MessagePart>) -> Self {
        if parts.len() == 1 {
            if let MessagePart::Text {
                text,
                cache_control: None,
            } = &parts[0]
            {
                return Self::Plain(text.clone());
            }
        }
        Self::Multipart(parts)
    }
}

impl From<String> for MessageContent {
    fn from(text: String) -> Self {
        Self::Plain(text)
    }
}

impl From<&str> for MessageContent {
    fn from(text: &str) -> Self {
        Self::Plain(text.to_string())
    }
}

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Debug, Serialize, Deserialize, Clone, Copy, Eq, PartialEq)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum CacheControl {
    Ephemeral {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        ttl: Option<CacheTtl>,
    },
}

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Debug, Serialize, Deserialize, Clone, Copy, Eq, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum CacheTtl {
    #[serde(rename = "1h")]
    OneHour,
}

#[derive(Serialize, Deserialize, Debug, Eq, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MessagePart {
    Text {
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    #[serde(rename = "image_url")]
    Image { image_url: String },
}

#[derive(Serialize, Deserialize, Debug, Eq, PartialEq)]
pub struct ToolCall {
    pub id: String,
    #[serde(flatten)]
    pub content: ToolCallContent,
}

#[derive(Serialize, Deserialize, Debug, Eq, PartialEq)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ToolCallContent {
    Function { function: FunctionContent },
}

#[derive(Serialize, Deserialize, Debug, Eq, PartialEq)]
pub struct FunctionContent {
    pub name: String,
    pub arguments: String,
}

#[derive(Serialize, Deserialize, Debug, Eq, PartialEq)]
pub struct ResponseMessageDelta {
    pub role: Option<Role>,
    pub content: Option<String>,
    pub reasoning_content: Option<String>,
    #[serde(default, skip_serializing_if = "is_none_or_empty")]
    pub tool_calls: Option<Vec<ToolCallChunk>>,
    /// Thinking blocks returned by Anthropic models via LiteLLM.
    /// Each block contains the full thinking text and its signature.
    /// Only populated on the final streaming chunk for a given thinking block.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub thinking_blocks: Vec<ThinkingBlock>,
    /// Compaction blocks returned when context compaction is active.
    /// Populated on the streaming chunk that contains the compaction summary.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_specific_fields: Option<ResponseProviderSpecificFields>,
}

#[derive(Serialize, Deserialize, Debug, Eq, PartialEq)]
pub struct ThinkingBlock {
    #[serde(rename = "type")]
    pub block_type: String,
    pub thinking: String,
    pub signature: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Eq, PartialEq)]
pub struct ToolCallChunk {
    pub index: usize,
    pub id: Option<String>,
    pub function: Option<FunctionChunk>,
}

#[derive(Serialize, Deserialize, Debug, Eq, PartialEq)]
pub struct FunctionChunk {
    pub name: Option<String>,
    pub arguments: Option<String>,
}

/// A compaction block. Used both in outgoing assistant messages (to replay a
/// stored summary) and in incoming streaming response deltas.
///
/// The `content` field is optional because the provider may stream it in
/// chunks; it is `None` on the initial block-start event.
/// `encrypted_content` is opaque metadata that Anthropic requires to be
/// round-tripped verbatim alongside the summary.
#[derive(Serialize, Deserialize, Debug, Eq, PartialEq)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum CompactionBlock {
    Compaction {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        content: Option<String>,
        /// Opaque metadata from a prior Anthropic compaction, forwarded
        /// verbatim by LiteLLM. `None` for non-Anthropic backends.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        encrypted_content: Option<String>,
    },
}

/// Wrapper for provider-specific fields in an outgoing assistant message.
/// Used to replay compaction blocks from a previous compacted response.
#[derive(Serialize, Deserialize, Debug, Eq, PartialEq)]
pub struct ProviderSpecificFields {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub compaction_blocks: Vec<CompactionBlock>,
}

/// Provider-specific fields returned in a streaming response delta.
#[derive(Serialize, Deserialize, Debug, Eq, PartialEq)]
pub struct ResponseProviderSpecificFields {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub compaction_blocks: Vec<CompactionBlock>,
}

/// Server-side context compaction settings sent in the request.
#[derive(Serialize, Deserialize, Debug, Eq, PartialEq)]
pub struct ContextManagement {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub edits: Vec<ContextManagementEdit>,
}

#[derive(Serialize, Deserialize, Debug, Eq, PartialEq)]
#[serde(tag = "type")]
pub enum ContextManagementEdit {
    #[serde(rename = "compact_20260112")]
    Compact {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        trigger: Option<CompactionTrigger>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pause_after_compaction: Option<bool>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        instructions: Option<String>,
    },
}

#[derive(Serialize, Deserialize, Debug, Eq, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CompactionTrigger {
    InputTokens { value: u64 },
}

#[derive(Serialize, Deserialize, Debug)]
pub struct PromptTokensDetails {
    #[serde(default)]
    pub cached_tokens: u64,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Usage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_tokens: Option<u64>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ChoiceDelta {
    pub index: u32,
    pub delta: ResponseMessageDelta,
    pub finish_reason: Option<String>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ResponseStreamEvent {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub created: u32,
    pub model: String,
    pub choices: Vec<ChoiceDelta>,
    pub usage: Option<Usage>,
}

/// Stream completions from LiteLLM using the OpenAI-compatible API.
/// Authenticates with `Authorization: Bearer {api_key}`.
pub async fn stream_completion(
    client: &dyn HttpClient,
    api_url: &str,
    api_key: &str,
    request: Request,
) -> Result<BoxStream<'static, Result<ResponseStreamEvent, LiteLlmError>>, LiteLlmError> {
    let uri = format!("{api_url}/v1/chat/completions");
    let request_builder = HttpRequest::builder()
        .method(Method::POST)
        .uri(uri)
        .header("Content-Type", "application/json")
        .header("Authorization", format!("Bearer {}", api_key));

    let body = serde_json::to_string(&request);
    let request = request_builder
        .body(AsyncBody::from(
            body.map_err(LiteLlmError::SerializeRequest)?,
        ))
        .map_err(LiteLlmError::BuildRequestBody)?;
    let host = request.uri().host().unwrap_or(api_url).to_owned();
    let mut response = client
        .send(request)
        .await
        .map_err(|error| LiteLlmError::HttpSend { host, error })?;

    if response.status().is_success() {
        let reader = BufReader::new(response.into_body());
        Ok(reader
            .lines()
            .filter_map(|line| async move {
                match line {
                    Ok(line) => {
                        if line.starts_with(':') {
                            return None;
                        }

                        let line = line.strip_prefix("data: ")?;
                        if line == "[DONE]" {
                            None
                        } else {
                            match serde_json::from_str::<ResponseStreamEvent>(line) {
                                Ok(response) => Some(Ok(response)),
                                Err(error) => {
                                    if line.trim().is_empty() {
                                        None
                                    } else {
                                        Some(Err(LiteLlmError::DeserializeResponse(error)))
                                    }
                                }
                            }
                        }
                    }
                    Err(error) => Some(Err(LiteLlmError::ReadResponse(error))),
                }
            })
            .boxed())
    } else {
        let mut body = String::new();
        response
            .body_mut()
            .read_to_string(&mut body)
            .await
            .map_err(LiteLlmError::ReadResponse)?;

        Err(error_from_response(response, body))
    }
}

// ---------- Error types ----------

#[derive(Debug)]
pub enum LiteLlmError {
    /// Failed to serialize the HTTP request body to JSON
    SerializeRequest(serde_json::Error),

    /// Failed to construct the HTTP request body
    BuildRequestBody(http::Error),

    /// Failed to send the HTTP request
    HttpSend { host: String, error: anyhow::Error },

    /// Failed to deserialize the response from JSON
    DeserializeResponse(serde_json::Error),

    /// Failed to read from response stream
    ReadResponse(io::Error),

    /// Rate limit exceeded
    RateLimit {
        status: http::StatusCode,
        retry_after: Duration,
    },

    /// Server overloaded
    ServerOverloaded {
        status: http::StatusCode,
        retry_after: Option<Duration>,
    },

    /// API returned an error response
    ApiError {
        status: http::StatusCode,
        message: String,
    },
}

/// Maps a non-successful LiteLLM response to a [`LiteLlmError`], preserving
/// the `Retry-After` header for rate-limit and overload responses.
fn error_from_response(response: http::Response<AsyncBody>, body: String) -> LiteLlmError {
    let status = response.status();
    let error_response = match serde_json::from_str::<LiteLlmErrorResponse>(&body) {
        Ok(LiteLlmErrorResponse { error }) => error,
        Err(_) => LiteLlmErrorBody {
            code: status.as_u16(),
            message: body,
            metadata: None,
        },
    };

    match status.as_u16() {
        // rate limited
        429 => {
            let retry_after = extract_retry_after(response.headers());
            LiteLlmError::RateLimit {
                status,
                retry_after: retry_after.unwrap_or_else(|| Duration::from_secs(60)),
            }
        }
        // overloaded
        503 => {
            let retry_after = extract_retry_after(response.headers());
            LiteLlmError::ServerOverloaded {
                status,
                retry_after,
            }
        }
        _ => LiteLlmError::ApiError {
            status,
            message: error_response.message,
        },
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LiteLlmErrorBody {
    pub code: u16,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<std::collections::HashMap<String, serde_json::Value>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LiteLlmErrorResponse {
    pub error: LiteLlmErrorBody,
}

impl From<LiteLlmError> for language_model_core::LanguageModelCompletionError {
    fn from(error: LiteLlmError) -> Self {
        use language_model_core::ProviderErrorCategory;
        let provider = language_model_core::LanguageModelProviderName::new("LiteLLM");
        match error {
            LiteLlmError::SerializeRequest(error) => Self::SerializeRequest { provider, error },
            LiteLlmError::BuildRequestBody(error) => Self::BuildRequestBody { provider, error },
            LiteLlmError::HttpSend { host, error } => Self::HttpSend {
                provider,
                host,
                error,
            },
            LiteLlmError::DeserializeResponse(error) => {
                Self::DeserializeResponse { provider, error }
            }
            LiteLlmError::ReadResponse(error) => Self::ApiReadResponseError { provider, error },
            LiteLlmError::RateLimit {
                status,
                retry_after,
            } => {
                let message = format!("{provider}'s API rate limit exceeded");
                Self::from_provider_response(
                    provider,
                    Some(status),
                    None,
                    message,
                    Some(retry_after),
                    ProviderErrorCategory::RateLimit,
                )
            }
            LiteLlmError::ServerOverloaded {
                status,
                retry_after,
            } => {
                let message = format!("{provider}'s API servers are overloaded right now");
                Self::from_provider_response(
                    provider,
                    Some(status),
                    None,
                    message,
                    retry_after,
                    ProviderErrorCategory::Overloaded,
                )
            }
            LiteLlmError::ApiError { status, message } => {
                let message = format!("{provider} API Error: {status}: {message}");
                Self::from_http_status(provider, status, message, None)
            }
        }
    }
}
