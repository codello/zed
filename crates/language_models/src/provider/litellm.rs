use anthropic::completion::{
    provider_compaction_encrypted_content, provider_compaction_state_from_encrypted_content,
};
use anyhow::Result;
use collections::HashMap;
use credentials_provider::CredentialsProvider;
use futures::{FutureExt, Stream, StreamExt, future::BoxFuture};
use gpui::{App, AppContext, AsyncApp, Context, Entity, SharedString, Task};
use http_client::HttpClient;
use language_model::{
    ANTHROPIC_PROVIDER_ID, ApiKeyConfiguration, ApiKeyState, AuthenticateError, CompactedContext,
    CompactionUpdate, EnvVar, IconOrSvg, LanguageModel, LanguageModelClient,
    LanguageModelCompletionError, LanguageModelCompletionEvent, LanguageModelCompletionStream,
    LanguageModelCostInfo, LanguageModelEffortLevel, LanguageModelId, LanguageModelName,
    LanguageModelProvider, LanguageModelProviderId, LanguageModelProviderName,
    LanguageModelProviderState, LanguageModelRequest, LanguageModelRequestToolInput,
    LanguageModelToolChoice, LanguageModelToolChoiceSupport, LanguageModelToolResultContent,
    LanguageModelToolUse, LanguageModelToolUseInput, MessageContent, ModelRateLimiters,
    ProviderSettingsView, Role, StopReason, TokenUsage, env_var, unavailable_error,
};
use litellm::{Dialect, ResponseStreamEvent, ThinkingDisplay, fetch_models};
use settings::{Settings, SettingsStore};
use std::pin::Pin;
use std::sync::{Arc, LazyLock};
use ui::IconName;
use util::ResultExt;

use language_model::util::{fix_streamed_json, parse_tool_arguments};

const PROVIDER_ID: LanguageModelProviderId = LanguageModelProviderId::new("litellm");
const PROVIDER_NAME: LanguageModelProviderName = LanguageModelProviderName::new("LiteLLM");

const API_KEY_ENV_VAR_NAME: &str = "LITELLM_API_KEY";
static API_KEY_ENV_VAR: LazyLock<EnvVar> = env_var!(API_KEY_ENV_VAR_NAME);

#[derive(Default, Clone, Debug, PartialEq)]
pub struct LiteLlmSettings {
    pub api_url: String,
}

pub struct LiteLlmLanguageModelProvider {
    http_client: Arc<dyn HttpClient>,
    state: Entity<State>,
    request_limiters: ModelRateLimiters,
}

pub struct State {
    api_key_state: ApiKeyState,
    credentials_provider: Arc<dyn CredentialsProvider>,
    http_client: Arc<dyn HttpClient>,
    available_models: Vec<litellm::Model>,
    fetch_models_task: Option<Task<Result<(), LanguageModelCompletionError>>>,
}

impl State {
    fn is_authenticated(&self) -> bool {
        self.api_key_state.has_key()
    }

    fn set_api_key(&mut self, api_key: Option<String>, cx: &mut Context<Self>) -> Task<Result<()>> {
        let credentials_provider = self.credentials_provider.clone();
        let api_url = LiteLlmLanguageModelProvider::api_url(cx);
        let task = self.api_key_state.store(
            api_url,
            api_key,
            |this| &mut this.api_key_state,
            credentials_provider,
            cx,
        );

        cx.spawn(async move |this, cx| {
            let result = task.await?;
            this.update(cx, |this, cx| this.restart_fetch_models_task(cx))
                .ok();
            Ok(result)
        })
    }

    fn authenticate(&mut self, cx: &mut Context<Self>) -> Task<Result<(), AuthenticateError>> {
        let credentials_provider = self.credentials_provider.clone();
        let api_url = LiteLlmLanguageModelProvider::api_url(cx);
        let task = self.api_key_state.load_if_needed(
            api_url,
            |this| &mut this.api_key_state,
            credentials_provider,
            cx,
        );

        cx.spawn(async move |this, cx| {
            let result = task.await;
            this.update(cx, |this, cx| this.restart_fetch_models_task(cx))
                .ok();
            result
        })
    }

    fn fetch_models(
        &mut self,
        cx: &mut Context<Self>,
    ) -> Task<Result<(), LanguageModelCompletionError>> {
        let http_client = self.http_client.clone();
        let api_url = LiteLlmLanguageModelProvider::api_url(cx);
        let api_key = self.api_key_state.key(&api_url);
        cx.spawn(async move |this, cx| {
            let models = fetch_models(http_client.as_ref(), &api_url, api_key.as_deref())
                .await
                .map_err(LanguageModelCompletionError::from)?;

            this.update(cx, |this, cx| {
                this.available_models = models;
                cx.notify();
            })
            .map_err(LanguageModelCompletionError::Other)?;

            Ok(())
        })
    }

    fn restart_fetch_models_task(&mut self, cx: &mut Context<Self>) {
        if self.is_authenticated() {
            let task = self.fetch_models(cx);
            self.fetch_models_task.replace(task);
        } else {
            self.available_models.clear();
        }
    }
}

impl LiteLlmLanguageModelProvider {
    pub fn new(
        http_client: Arc<dyn HttpClient>,
        credentials_provider: Arc<dyn CredentialsProvider>,
        cx: &mut App,
    ) -> Self {
        let state = cx.new(|cx| {
            cx.observe_global::<SettingsStore>({
                let mut last_settings = LiteLlmLanguageModelProvider::settings(cx).clone();
                move |this: &mut State, cx| {
                    let current_settings = LiteLlmLanguageModelProvider::settings(cx);
                    let settings_changed = current_settings != &last_settings;
                    if settings_changed {
                        last_settings = current_settings.clone();
                        this.authenticate(cx).detach();
                        cx.notify();
                    }
                }
            })
            .detach();
            State {
                api_key_state: ApiKeyState::new(Self::api_url(cx), (*API_KEY_ENV_VAR).clone()),
                credentials_provider,
                http_client: http_client.clone(),
                available_models: Vec::new(),
                fetch_models_task: None,
            }
        });

        Self {
            http_client,
            state,
            request_limiters: ModelRateLimiters::default(),
        }
    }

    fn settings(cx: &App) -> &LiteLlmSettings {
        &crate::AllLanguageModelSettings::get_global(cx).litellm
    }

    fn api_url(cx: &App) -> SharedString {
        let api_url = &Self::settings(cx).api_url;
        if api_url.is_empty() {
            litellm::LITELLM_API_URL.into()
        } else {
            SharedString::new(api_url.as_str())
        }
    }

    /// The current configuration of `model`, if this provider still offers it.
    fn config(
        &self,
        model: &LanguageModel,
        cx: &App,
    ) -> Result<litellm::Model, LanguageModelCompletionError> {
        self.state
            .read(cx)
            .available_models
            .iter()
            .find(|listed| listed.name == model.id.0.as_ref())
            .cloned()
            .ok_or_else(|| unavailable_error(model))
    }

    fn stream_litellm_request(
        &self,
        request: litellm::Request,
        cx: &AsyncApp,
    ) -> BoxFuture<
        'static,
        Result<
            futures::stream::BoxStream<'static, Result<ResponseStreamEvent, litellm::LiteLlmError>>,
            LanguageModelCompletionError,
        >,
    > {
        let http_client = self.http_client.clone();
        let (api_key, api_url) = self.state.read_with(cx, |state, cx| {
            let api_url = LiteLlmLanguageModelProvider::api_url(cx);
            (state.api_key_state.key(&api_url), api_url)
        });

        async move {
            let Some(api_key) = api_key else {
                return Err(LanguageModelCompletionError::NoApiKey {
                    provider: PROVIDER_NAME,
                });
            };
            litellm::stream_completion(http_client.as_ref(), &api_url, &api_key, request)
                .await
                .map_err(Into::into)
        }
        .boxed()
    }
}

impl LanguageModelProviderState for LiteLlmLanguageModelProvider {
    type ObservableEntity = State;

    fn observable_entity(&self) -> Option<Entity<Self::ObservableEntity>> {
        Some(self.state.clone())
    }
}

impl LanguageModelProvider for LiteLlmLanguageModelProvider {
    fn id(&self) -> LanguageModelProviderId {
        PROVIDER_ID
    }

    fn name(&self) -> LanguageModelProviderName {
        PROVIDER_NAME
    }

    fn icon(&self) -> IconOrSvg {
        IconOrSvg::Icon(IconName::AiMaibornWolff)
    }

    fn default_model(&self, _cx: &App) -> Option<LanguageModel> {
        None
    }

    fn default_fast_model(&self, _cx: &App) -> Option<LanguageModel> {
        None
    }

    fn provided_models(&self, cx: &App) -> Vec<LanguageModel> {
        self.state
            .read(cx)
            .available_models
            .iter()
            .map(language_model)
            .collect()
    }

    fn is_authenticated(&self, cx: &App) -> bool {
        self.state.read(cx).is_authenticated()
    }

    fn authenticate(&self, cx: &mut App) -> Task<Result<(), AuthenticateError>> {
        self.state.update(cx, |state, cx| state.authenticate(cx))
    }

    fn settings_view(&self, cx: &mut App) -> Option<ProviderSettingsView> {
        let state = self.state.read(cx);
        Some(ProviderSettingsView::ApiKey(ApiKeyConfiguration::new(
            state.api_key_state.has_key(),
            state.api_key_state.is_from_env_var(),
            state.api_key_state.env_var_name().clone(),
            "https://docs.litellm.ai/".into(),
        )))
    }

    fn set_api_key(&self, api_key: Option<String>, cx: &mut App) -> Task<Result<()>> {
        self.state
            .update(cx, |state, cx| state.set_api_key(api_key, cx))
    }
}

impl LanguageModelClient for LiteLlmLanguageModelProvider {
    fn stream_completion(
        &self,
        model: &LanguageModel,
        request: LanguageModelRequest,
        cx: &AsyncApp,
    ) -> BoxFuture<'static, Result<LanguageModelCompletionStream, LanguageModelCompletionError>>
    {
        let config = match cx.update(|cx| self.config(model, cx)) {
            Ok(config) => config,
            Err(error) => return async move { Err(error) }.boxed(),
        };
        let litellm_request = match into_litellm(request, &config) {
            Ok(request) => request,
            Err(error) => return async move { Err(error.into()) }.boxed(),
        };
        let request = self.stream_litellm_request(litellm_request, cx);
        let request_limiter = self.request_limiters.for_model(&model.id);
        let executor = cx.background_executor().clone();
        let future = request_limiter.stream(async move {
            let response = request.await?;
            let events = LiteLlmEventMapper::new().map_stream(response);
            Ok(language_model::stream_in_background(
                events.boxed(),
                executor,
            ))
        });
        async move { Ok(future.await?.boxed()) }.boxed()
    }

    fn api_key(&self, _model: &LanguageModel, cx: &App) -> Option<String> {
        self.state.read_with(cx, |state, cx| {
            let api_url = LiteLlmLanguageModelProvider::api_url(cx);
            state.api_key_state.key(&api_url).map(|key| key.to_string())
        })
    }
}

fn language_model(model: &litellm::Model) -> LanguageModel {
    let reasoning_efforts = model.reasoning_efforts();
    let default_effort = model.default_reasoning_effort();
    LanguageModel {
        cost_info: model
            .input_cost_per_token
            .zip(model.output_cost_per_token)
            // The proxy reports zero for both costs when it has no pricing
            // data for a model; treat that as unknown.
            .filter(|(input_cost, output_cost)| *input_cost != 0.0 || *output_cost != 0.0)
            .map(
                |(input_cost, output_cost)| LanguageModelCostInfo::TokenCost {
                    input_token_cost_per_1m: 1_000_000.0 * input_cost,
                    output_token_cost_per_1m: 1_000_000.0 * output_cost,
                },
            ),
        supports_tools: model.supports_tools,
        supports_streaming_tools: model.supports_tools,
        supports_thinking: model.supports_thinking || !reasoning_efforts.is_empty(),
        supported_effort_levels: reasoning_efforts
            .iter()
            .map(|&effort| LanguageModelEffortLevel {
                name: effort.label().into(),
                value: effort.value().into(),
                is_default: default_effort == Some(effort),
            })
            .collect(),
        tool_choice_support: if model.supports_tool_choice {
            LanguageModelToolChoiceSupport::ALL
        } else {
            LanguageModelToolChoiceSupport::default()
        },
        supports_images: model.supports_vision,
        supports_split_token_display: true,
        supports_server_side_compaction: model.supports_context_management,
        max_output_tokens: model.max_output_tokens(),
        ..LanguageModel::new(
            LanguageModelId::from(model.name.clone()),
            LanguageModelName::from(model.display_name().to_string()),
            PROVIDER_ID,
            PROVIDER_NAME,
            format!("litellm/{}", model.name),
            model.max_token_count(),
        )
    }
}

pub fn into_litellm(
    request: LanguageModelRequest,
    model: &litellm::Model,
) -> Result<litellm::Request> {
    let max_tokens = request.effective_max_output_tokens(model.max_output_tokens());
    if request.contains_custom_tool_input() {
        anyhow::bail!("LiteLLM does not support custom tools");
    }

    let mut messages: Vec<litellm::RequestMessage> = Vec::new();
    let supports_cache_control = model.dialect() == Some(Dialect::Anthropic);
    let mut any_message_wants_cache = false;
    let reasoning_efforts = model.reasoning_efforts();

    for message in request.messages {
        if message.contents_empty() {
            continue;
        }
        any_message_wants_cache |= message.cache;

        let mut parts: Vec<litellm::MessagePart> = Vec::new();
        let mut tool_calls: Vec<litellm::ToolCall> = Vec::new();
        let mut thinking_blocks: Vec<litellm::ThinkingBlock> = Vec::new();
        let mut compaction_blocks: Vec<litellm::CompactionBlock> = Vec::new();

        for content in message.content {
            match content {
                MessageContent::Text(text) => {
                    parts.push(litellm::MessagePart::Text {
                        text,
                        cache_control: None,
                    });
                }
                MessageContent::Thinking { text, signature } => {
                    if let Some(signature) = signature
                        && !signature.is_empty()
                    {
                        thinking_blocks.push(litellm::ThinkingBlock {
                            block_type: "thinking".to_string(),
                            thinking: text,
                            signature: Some(signature),
                        });
                    }
                }
                MessageContent::RedactedThinking(_) => {}
                MessageContent::Compaction(CompactedContext::Summary {
                    content,
                    provider_state,
                }) => {
                    // Replay the compaction summary block in the assistant message via
                    // `provider_specific_fields`, which LiteLLM forwards to Anthropic
                    // (and other supported backends).
                    compaction_blocks.push(litellm::CompactionBlock::Compaction {
                        content: Some(content.to_string()),
                        encrypted_content: provider_state
                            .map(|state| {
                                provider_compaction_encrypted_content(
                                    &state,
                                    &ANTHROPIC_PROVIDER_ID,
                                )
                                .log_err()
                                .flatten()
                                .map(|content| content.to_string())
                            })
                            .flatten(),
                    });
                }
                // A standalone `ProviderState` carries no textual summary to
                // replay, and only its owning backend knows how to interpret it;
                // drop it here (mirrors the native Anthropic provider).
                MessageContent::Compaction(CompactedContext::ProviderState(_)) => {}
                MessageContent::Image(image) => {
                    parts.push(litellm::MessagePart::Image {
                        image_url: image.to_base64_url(),
                    });
                }
                MessageContent::ToolUse(tool_use) => {
                    tool_calls.push(litellm::ToolCall {
                        id: tool_use.id.to_string(),
                        content: litellm::ToolCallContent::Function {
                            function: litellm::FunctionContent {
                                name: tool_use.name.to_string(),
                                arguments: match &tool_use.input {
                                    LanguageModelToolUseInput::Json(v) => {
                                        serde_json::to_string(v).unwrap_or_default()
                                    }
                                    LanguageModelToolUseInput::Text(t) => t.to_string(),
                                },
                            },
                        },
                    });
                }
                MessageContent::ToolResult(tool_result) => {
                    let content: Vec<litellm::MessagePart> = tool_result
                        .content
                        .into_iter()
                        .map(|part| match part {
                            LanguageModelToolResultContent::Text(text) => {
                                litellm::MessagePart::Text {
                                    text: text.to_string(),
                                    cache_control: None,
                                }
                            }
                            LanguageModelToolResultContent::Image(image) => {
                                litellm::MessagePart::Image {
                                    image_url: image.to_base64_url(),
                                }
                            }
                        })
                        .collect();

                    messages.push(litellm::RequestMessage::Tool {
                        content: content.into(),
                        tool_call_id: tool_result.tool_use_id.to_string(),
                    });
                }
            }
        }

        let provider_specific_fields = (!compaction_blocks.is_empty())
            .then_some(litellm::ProviderSpecificFields { compaction_blocks });

        if parts.is_empty()
            && tool_calls.is_empty()
            && thinking_blocks.is_empty()
            && provider_specific_fields.is_none()
        {
            continue;
        } else if let Some(litellm::RequestMessage::User { content }) = messages.last_mut()
            && message.role == Role::User
        {
            for part in parts {
                content.push_part(part);
            }
        } else if let Some(litellm::RequestMessage::System { content }) = messages.last_mut()
            && message.role == Role::System
        {
            for part in parts {
                content.push_part(part);
            }
        } else if let Some(litellm::RequestMessage::Assistant {
            content: last_content,
            tool_calls: last_tool_calls,
            thinking_blocks: last_thinking_blocks,
            provider_specific_fields: last_provider_specific_fields,
        }) = messages.last_mut()
            && message.role == Role::Assistant
        {
            if let Some(last_content) = last_content {
                for part in parts {
                    last_content.push_part(part);
                }
            } else if !parts.is_empty() {
                *last_content = Some(litellm::MessageContent::from(parts));
            }
            last_tool_calls.extend(tool_calls);
            last_thinking_blocks.extend(thinking_blocks);
            if let Some(fields) = provider_specific_fields {
                last_provider_specific_fields
                    .get_or_insert_with(|| litellm::ProviderSpecificFields {
                        compaction_blocks: Vec::new(),
                    })
                    .compaction_blocks
                    .extend(fields.compaction_blocks);
            }
        } else {
            messages.push(match message.role {
                Role::User => litellm::RequestMessage::User {
                    content: litellm::MessageContent::from(parts),
                },
                Role::System => litellm::RequestMessage::System {
                    content: litellm::MessageContent::from(parts),
                },
                Role::Assistant => litellm::RequestMessage::Assistant {
                    content: (!parts.is_empty()).then_some(litellm::MessageContent::from(parts)),
                    tool_calls,
                    thinking_blocks,
                    provider_specific_fields,
                },
            })
        }
    }

    // When any message requests caching, mark the stable prefix (tools and
    // system prompt) with a long-TTL breakpoint, and use the top-level
    // cache_control to let Anthropic automatically handle the short-TTL
    // conversation breakpoint. The prefix order Anthropic uses is
    // tools → system → messages, so long-TTL entries must come first.
    // LiteLLM forwards both the per-block cache_control and the top-level
    // cache_control to Anthropic. Non-Anthropic backends reject these fields,
    // so we only include them when the model dialect is Anthropic.
    let long_lived_cache = (any_message_wants_cache && supports_cache_control).then_some(
        litellm::CacheControl::Ephemeral {
            ttl: Some(litellm::CacheTtl::OneHour),
        },
    );

    // Build the tools list, marking the last tool with a long-TTL breakpoint.
    let mut tools: Vec<litellm::ToolDefinition> = request
        .tools
        .into_iter()
        .map(|tool| {
            let LanguageModelRequestToolInput::Function { input_schema, .. } = tool.input else {
                return Err(anyhow::anyhow!("LiteLLM does not support custom tools"));
            };
            Ok(litellm::ToolDefinition::Function {
                function: litellm::FunctionDefinition {
                    name: tool.name,
                    description: Some(tool.description),
                    parameters: Some(input_schema),
                },
                cache_control: None,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    if let Some(litellm::ToolDefinition::Function { cache_control, .. }) = tools.last_mut() {
        *cache_control = long_lived_cache;
    }

    // Mark the system prompt with a long-TTL breakpoint.
    if let Some(content) = messages.iter_mut().rev().find_map(|message| match message {
        litellm::RequestMessage::System { content } => Some(content),
        _ => None,
    }) {
        set_last_text_cache_control(content, long_lived_cache);
    }

    Ok(litellm::Request {
        model: model.name.clone(),
        messages,
        stream: true,
        stream_options: Some(litellm::StreamOptions {
            include_usage: true,
        }),
        stop: request.stop,
        temperature: request.temperature,
        max_tokens,
        parallel_tool_calls: (model.supports_parallel_tool_calls && !tools.is_empty())
            .then_some(true),
        reasoning_effort: request
            .thinking_allowed
            .then(|| {
                request.thinking_effort.or_else(|| {
                    model
                        .default_reasoning_effort()
                        .map(|effort| effort.value().to_string())
                })
            })
            .flatten(),
        thinking: (request.thinking_allowed && model.supports_thinking).then_some(
            if !reasoning_efforts.is_empty() {
                litellm::Thinking::Adaptive {
                    display: Some(ThinkingDisplay::Summarized),
                }
            } else {
                litellm::Thinking::Manual {
                    budget_tokens: 4_096,
                }
            },
        ),
        tools,
        tool_choice: request.tool_choice.map(|choice| match choice {
            LanguageModelToolChoice::Auto => litellm::ToolChoice::Auto,
            LanguageModelToolChoice::Any => litellm::ToolChoice::Required,
            LanguageModelToolChoice::None => litellm::ToolChoice::None,
        }),
        // Top-level automatic caching for the conversation tail (short TTL).
        // Anthropic places the breakpoint on the last cacheable block automatically.
        // Only sent for Anthropic-dialect models; other backends reject this field.
        cache_control: (any_message_wants_cache && supports_cache_control)
            .then_some(litellm::CacheControl::Ephemeral { ttl: None }),
        context_management: (model.supports_context_management)
            .then(|| request.compact_at_tokens)
            .flatten()
            .map(|value| litellm::ContextManagement {
                edits: vec![litellm::ContextManagementEdit::Compact {
                    trigger: Some(litellm::CompactionTrigger::InputTokens { value }),
                    pause_after_compaction: None,
                    instructions: None,
                }],
            }),
    })
}

fn set_last_text_cache_control(
    content: &mut litellm::MessageContent,
    cache_control: Option<litellm::CacheControl>,
) {
    let Some(cache_control) = cache_control else {
        return;
    };
    match content {
        litellm::MessageContent::Plain(text) => {
            let text = std::mem::take(text);
            *content = litellm::MessageContent::Multipart(vec![litellm::MessagePart::Text {
                text,
                cache_control: Some(cache_control),
            }]);
        }
        litellm::MessageContent::Multipart(parts) => {
            for part in parts.iter_mut().rev() {
                if let litellm::MessagePart::Text {
                    cache_control: target,
                    ..
                } = part
                {
                    *target = Some(cache_control);
                    break;
                }
            }
        }
    }
}

pub struct LiteLlmEventMapper {
    tool_calls_by_index: HashMap<usize, RawToolCall>,
    /// Accumulated thinking signatures from thinking_blocks in the streaming response.
    /// Keyed by the thinking text to match signatures to their blocks.
    thinking_signatures: Vec<String>,
}

impl LiteLlmEventMapper {
    pub fn new() -> Self {
        Self {
            tool_calls_by_index: HashMap::default(),
            thinking_signatures: Vec::new(),
        }
    }

    pub fn map_stream(
        mut self,
        events: Pin<
            Box<dyn Send + Stream<Item = Result<ResponseStreamEvent, litellm::LiteLlmError>>>,
        >,
    ) -> impl Stream<Item = Result<LanguageModelCompletionEvent, LanguageModelCompletionError>>
    {
        events.flat_map(move |event| {
            futures::stream::iter(match event {
                Ok(event) => self.map_event(event),
                Err(error) => vec![Err(error.into())],
            })
        })
    }

    pub fn map_event(
        &mut self,
        event: ResponseStreamEvent,
    ) -> Vec<Result<LanguageModelCompletionEvent, LanguageModelCompletionError>> {
        let mut events = Vec::new();

        if let Some(usage) = event.usage {
            let cache_read_input_tokens = usage
                .prompt_tokens_details
                .as_ref()
                .map_or(0, |details| details.cached_tokens);
            let input_tokens = usage.prompt_tokens.saturating_sub(cache_read_input_tokens);

            events.push(Ok(LanguageModelCompletionEvent::UsageUpdate(TokenUsage {
                input_tokens,
                output_tokens: usage.completion_tokens,
                cache_creation_input_tokens: usage.cache_creation_input_tokens.unwrap_or(0),
                cache_read_input_tokens,
            })));
        }

        let Some(choice) = event.choices.first() else {
            return events;
        };

        // Capture signatures from thinking_blocks before emitting thinking text.
        // For Anthropic models via LiteLLM, the signature arrives in a final chunk
        // alongside the complete thinking_blocks data.
        for block in &choice.delta.thinking_blocks {
            if let Some(signature) = block.signature.clone()
                && !signature.is_empty()
            {
                self.thinking_signatures.push(signature.clone());
            }
        }

        if let Some(text) = choice.delta.reasoning_content.clone() {
            events.push(Ok(LanguageModelCompletionEvent::Thinking {
                text,
                signature: None,
            }));
        }

        // Emit any collected signatures as empty-text Thinking events, matching
        // the pattern used by the native Anthropic provider's SignatureDelta handling.
        for signature in self.thinking_signatures.drain(..) {
            events.push(Ok(LanguageModelCompletionEvent::Thinking {
                text: String::new(),
                signature: Some(signature),
            }));
        }

        if let Some(content) = choice.delta.content.clone() {
            if !content.is_empty() {
                events.push(Ok(LanguageModelCompletionEvent::Text(content)));
            }
        }

        // Emit compaction events from provider_specific_fields in the streaming delta.
        // Each `CompactionBlock` arrives whole (not as streamed deltas), so we mirror
        // the native Anthropic provider's per-block sequence: `Started`, then
        // `SummaryDelta` when there is content to surface, then `Finished` carrying
        // the persistable summary. An empty block is treated as a failed compaction.
        //
        // LiteLLM forwards Anthropic's raw compaction block, which may include
        // `encrypted_content`; when present we package it as a `ProviderCompactionState`
        // owned by the Anthropic backend so it can be round-tripped on the next request.
        for block in choice
            .delta
            .provider_specific_fields
            .as_ref()
            .map_or(&[][..], |f| f.compaction_blocks.as_slice())
        {
            match block {
                litellm::CompactionBlock::Compaction {
                    content,
                    encrypted_content,
                } => {
                    events.push(Ok(LanguageModelCompletionEvent::Compaction(
                        CompactionUpdate::Started,
                    )));
                    match content.as_deref().filter(|content| !content.is_empty()) {
                        Some(content) => {
                            events.push(Ok(LanguageModelCompletionEvent::Compaction(
                                CompactionUpdate::SummaryDelta(Arc::from(content)),
                            )));
                            events.push(Ok(LanguageModelCompletionEvent::Compaction(
                                CompactionUpdate::Finished(CompactedContext::Summary {
                                    content: Arc::from(content),
                                    provider_state: encrypted_content
                                        .as_deref()
                                        .filter(|encrypted| !encrypted.is_empty())
                                        .map(|encrypted| {
                                            provider_compaction_state_from_encrypted_content(
                                                ANTHROPIC_PROVIDER_ID.clone(),
                                                Arc::from(encrypted),
                                            )
                                        }),
                                }),
                            )));
                        }
                        None => {
                            events.push(Ok(LanguageModelCompletionEvent::Compaction(
                                CompactionUpdate::Failed,
                            )));
                        }
                    }
                }
            }
        }

        if let Some(tool_calls) = choice.delta.tool_calls.as_ref() {
            for tool_call in tool_calls {
                let entry = self.tool_calls_by_index.entry(tool_call.index).or_default();

                if let Some(tool_id) = tool_call.id.clone() {
                    entry.id = tool_id;
                }

                if let Some(function) = tool_call.function.as_ref() {
                    if let Some(name) = function.name.clone() {
                        entry.name = name;
                    }

                    if let Some(arguments) = function.arguments.clone() {
                        entry.arguments.push_str(&arguments);
                    }
                }

                if !entry.id.is_empty() && !entry.name.is_empty() {
                    if let Ok(value) = serde_json::from_str::<serde_json::Value>(
                        &fix_streamed_json(&entry.arguments),
                    ) {
                        events.push(Ok(LanguageModelCompletionEvent::ToolUse(
                            LanguageModelToolUse {
                                id: entry.id.clone().into(),
                                name: entry.name.as_str().into(),
                                is_input_complete: false,
                                input: LanguageModelToolUseInput::Json(value),
                                raw_input: entry.arguments.clone(),
                                thought_signature: None,
                            },
                        )));
                    }
                }
            }
        }

        match choice.finish_reason.as_deref() {
            Some("stop") => {
                events.push(Ok(LanguageModelCompletionEvent::Stop(StopReason::EndTurn)));
            }
            Some("tool_calls") => {
                events.extend(self.tool_calls_by_index.drain().map(|(_, tool_call)| {
                    match parse_tool_arguments(&tool_call.arguments) {
                        Ok(input) => Ok(LanguageModelCompletionEvent::ToolUse(
                            LanguageModelToolUse {
                                id: tool_call.id.clone().into(),
                                name: tool_call.name.as_str().into(),
                                is_input_complete: true,
                                input: LanguageModelToolUseInput::Json(input),
                                raw_input: tool_call.arguments.clone(),
                                thought_signature: None,
                            },
                        )),
                        Err(error) => Ok(LanguageModelCompletionEvent::ToolUseJsonParseError {
                            id: tool_call.id.clone().into(),
                            tool_name: tool_call.name.as_str().into(),
                            raw_input: tool_call.arguments.clone().into(),
                            json_parse_error: error.to_string(),
                        }),
                    }
                }));
                events.push(Ok(LanguageModelCompletionEvent::Stop(StopReason::ToolUse)));
            }
            Some("length") => {
                events.push(Ok(LanguageModelCompletionEvent::Stop(
                    StopReason::MaxTokens,
                )));
            }
            Some(stop_reason) => {
                log::error!("Unexpected LiteLLM stop_reason: {stop_reason:?}");
                events.push(Ok(LanguageModelCompletionEvent::Stop(StopReason::EndTurn)));
            }
            None => {}
        }

        events
    }
}

#[derive(Default)]
struct RawToolCall {
    id: String,
    name: String,
    arguments: String,
}
