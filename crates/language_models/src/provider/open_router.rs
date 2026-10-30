use anyhow::Result;

use credentials_provider::CredentialsProvider;
use futures::{FutureExt, StreamExt, future::BoxFuture};
use gpui::{App, AppContext, AsyncApp, Context, Entity, SharedString, Task};
use http_client::{CustomHeaders, HttpClient};
use language_model::chat_completion::ChatCompletionEventMapper;
use language_model::{
    ApiKeyConfiguration, ApiKeyState, AuthenticateError, EnvVar, IconOrSvg, LanguageModel,
    LanguageModelClient, LanguageModelCompletionError, LanguageModelCompletionStream,
    LanguageModelEffortLevel, LanguageModelId, LanguageModelName, LanguageModelProvider,
    LanguageModelProviderId, LanguageModelProviderName, LanguageModelProviderState,
    LanguageModelRequest, LanguageModelToolChoice, LanguageModelToolChoiceSupport,
    LanguageModelToolResultContent, MessageContent, ModelRateLimiters, ProviderSettingsView, Role,
    UpstreamProvider, UpstreamProviderPricing, env_var, unavailable_error,
};
use open_router::{
    Model, ModelMode as OpenRouterModelMode, OPEN_ROUTER_API_URL, ProviderEndpoint,
    ReasoningEffort, ResponseStreamEvent, list_models, list_provider_endpoints,
};
use scheduler::Instant;
use settings::{OpenRouterAvailableModel as AvailableModel, Settings, SettingsStore};
use sha2::{Digest as _, Sha256};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::{Arc, LazyLock};
use std::time::Duration;
use ui::IconName;
use util::ResultExt;

const PROVIDER_ID: LanguageModelProviderId = LanguageModelProviderId::new("openrouter");
const PROVIDER_NAME: LanguageModelProviderName = LanguageModelProviderName::new("OpenRouter");

const API_KEY_ENV_VAR_NAME: &str = "OPENROUTER_API_KEY";
const PROVIDER_ENDPOINTS_TTL: Duration = Duration::from_secs(60 * 60);
static API_KEY_ENV_VAR: LazyLock<EnvVar> = env_var!(API_KEY_ENV_VAR_NAME);
pub(crate) const RESERVED_HEADER_NAMES: &[&str] = &["HTTP-Referer", "X-Title"];

#[derive(Default, Clone, Debug, PartialEq)]
pub struct OpenRouterSettings {
    pub api_url: String,
    pub available_models: Vec<AvailableModel>,
    pub custom_headers: CustomHeaders,
}

pub struct OpenRouterLanguageModelProvider {
    http_client: Arc<dyn HttpClient>,
    state: Entity<State>,
    request_limiters: ModelRateLimiters,
}

pub struct State {
    api_key_state: ApiKeyState,
    credentials_provider: Arc<dyn CredentialsProvider>,
    http_client: Arc<dyn HttpClient>,
    available_models: Vec<open_router::Model>,
    upstream_providers: HashMap<String, CachedUpstreamProviders>,
    fetch_models_task: Option<Task<()>>,
    last_fetch_models_error: Option<String>,
}

struct CachedUpstreamProviders {
    providers: Vec<UpstreamProvider>,
    fetched_at: Instant,
}

impl State {
    fn is_authenticated(&self) -> bool {
        self.api_key_state.has_key()
    }

    fn set_api_key(&mut self, api_key: Option<String>, cx: &mut Context<Self>) -> Task<Result<()>> {
        let credentials_provider = self.credentials_provider.clone();
        let api_url = OpenRouterLanguageModelProvider::api_url(cx);
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
        let api_url = OpenRouterLanguageModelProvider::api_url(cx);
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
        let api_url = OpenRouterLanguageModelProvider::api_url(cx);
        let extra_headers = OpenRouterLanguageModelProvider::settings(cx)
            .custom_headers
            .clone();
        let Some(api_key) = self.api_key_state.key(&api_url) else {
            return Task::ready(Err(LanguageModelCompletionError::NoApiKey {
                provider: PROVIDER_NAME,
            }));
        };
        cx.spawn(async move |this, cx| {
            let models = list_models(http_client.as_ref(), &api_url, &api_key, &extra_headers)
                .await
                .map_err(LanguageModelCompletionError::from)?;

            this.update(cx, |this, cx| {
                this.available_models = models;
                cx.notify();
            })
            .map_err(|e| LanguageModelCompletionError::Other(e))?;

            Ok(())
        })
    }

    fn restart_fetch_models_task(&mut self, cx: &mut Context<Self>) {
        self.upstream_providers.clear();
        if self.is_authenticated() {
            let fetch_task = self.fetch_models(cx);
            let task = cx.spawn(async move |this, cx| {
                let result = fetch_task.await;
                this.update(cx, |this, _cx| {
                    this.last_fetch_models_error =
                        result.as_ref().err().map(|error| format!("{error:#}"));
                })
                .ok();
                result.log_err();
            });
            self.fetch_models_task.replace(task);
        } else {
            self.available_models.clear();
            self.last_fetch_models_error = None;
        }
    }
}

impl OpenRouterLanguageModelProvider {
    pub fn new(
        http_client: Arc<dyn HttpClient>,
        credentials_provider: Arc<dyn CredentialsProvider>,
        cx: &mut App,
    ) -> Self {
        let state = cx.new(|cx| {
            cx.observe_global::<SettingsStore>({
                let mut last_settings = OpenRouterLanguageModelProvider::settings(cx).clone();
                move |this: &mut State, cx| {
                    let current_settings = OpenRouterLanguageModelProvider::settings(cx);
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
                upstream_providers: HashMap::default(),
                fetch_models_task: None,
                last_fetch_models_error: None,
            }
        });

        Self {
            http_client,
            state,
            request_limiters: ModelRateLimiters::default(),
        }
    }

    fn settings(cx: &App) -> &OpenRouterSettings {
        &crate::AllLanguageModelSettings::get_global(cx).open_router
    }

    fn api_url(cx: &App) -> SharedString {
        let api_url = &Self::settings(cx).api_url;
        if api_url.is_empty() {
            OPEN_ROUTER_API_URL.into()
        } else {
            SharedString::new(api_url.as_str())
        }
    }

    /// Every model this provider lists, in order: the fetched list, with
    /// settings entries replacing fetched ones of the same name or appended.
    fn open_router_models(&self, cx: &App) -> Vec<open_router::Model> {
        let mut models_from_api = self.state.read(cx).available_models.clone();
        let mut settings_models = Vec::new();

        for model in &Self::settings(cx).available_models {
            let mode = model.mode.unwrap_or_default();
            let (supported_efforts, default_effort) =
                if matches!(mode, OpenRouterModelMode::Adaptive) {
                    (
                        model.reasoning_effort.into_iter().collect(),
                        model.reasoning_effort,
                    )
                } else {
                    (Vec::new(), None)
                };
            settings_models.push(open_router::Model {
                name: model.name.clone(),
                display_name: model.display_name.clone(),
                max_tokens: model.max_tokens,
                supports_tools: model.supports_tools,
                supports_images: model.supports_images,
                mode,
                supported_efforts,
                default_effort,
                supports_max_tokens: false,
                mandatory_reasoning: false,
                provider: model.provider.clone(),
            });
        }

        for settings_model in &settings_models {
            if let Some(pos) = models_from_api
                .iter()
                .position(|m| m.name == settings_model.name)
            {
                models_from_api[pos] = settings_model.clone();
            } else {
                models_from_api.push(settings_model.clone());
            }
        }

        models_from_api
    }

    /// The listed model with `id`, or the built-in default model when `id`
    /// names it, even if the fetched list doesn't include it.
    fn open_router_model(&self, id: &str, cx: &App) -> Option<open_router::Model> {
        self.open_router_models(cx)
            .into_iter()
            .find(|listed| listed.id() == id)
            .or_else(|| {
                let default = open_router::Model::default();
                (default.id() == id).then_some(default)
            })
    }

    /// The current configuration of `model`, if this provider still offers it.
    fn config(
        &self,
        model: &LanguageModel,
        cx: &App,
    ) -> Result<open_router::Model, LanguageModelCompletionError> {
        self.open_router_model(model.id.0.as_ref(), cx)
            .ok_or_else(|| unavailable_error(model))
    }

    fn stream_open_router_request(
        &self,
        request: open_router::Request,
        cx: &AsyncApp,
    ) -> BoxFuture<
        'static,
        Result<
            futures::stream::BoxStream<
                'static,
                Result<ResponseStreamEvent, open_router::OpenRouterError>,
            >,
            LanguageModelCompletionError,
        >,
    > {
        let http_client = self.http_client.clone();
        let (api_key, api_url, extra_headers) = self.state.read_with(cx, |state, cx| {
            let api_url = OpenRouterLanguageModelProvider::api_url(cx);
            let extra_headers = OpenRouterLanguageModelProvider::settings(cx)
                .custom_headers
                .clone();
            (state.api_key_state.key(&api_url), api_url, extra_headers)
        });

        async move {
            let Some(api_key) = api_key else {
                return Err(LanguageModelCompletionError::NoApiKey {
                    provider: PROVIDER_NAME,
                });
            };
            let request = open_router::stream_completion(
                http_client.as_ref(),
                &api_url,
                &api_key,
                request,
                &extra_headers,
            );
            request.await.map_err(Into::into)
        }
        .boxed()
    }
}

impl LanguageModelProviderState for OpenRouterLanguageModelProvider {
    type ObservableEntity = State;

    fn observable_entity(&self) -> Option<Entity<Self::ObservableEntity>> {
        Some(self.state.clone())
    }
}

impl LanguageModelProvider for OpenRouterLanguageModelProvider {
    fn id(&self) -> LanguageModelProviderId {
        PROVIDER_ID
    }

    fn name(&self) -> LanguageModelProviderName {
        PROVIDER_NAME
    }

    fn icon(&self) -> IconOrSvg {
        IconOrSvg::Icon(IconName::AiOpenRouter)
    }

    fn default_model(&self, cx: &App) -> Option<LanguageModel> {
        let default = open_router::Model::default();
        self.open_router_model(default.id(), cx)
            .map(|model| language_model(&model))
    }

    fn default_fast_model(&self, _cx: &App) -> Option<LanguageModel> {
        None
    }

    fn provided_models(&self, cx: &App) -> Vec<LanguageModel> {
        self.open_router_models(cx)
            .iter()
            .map(language_model)
            .collect()
    }

    fn upstream_providers(
        &self,
        model: &LanguageModel,
        cx: &mut App,
    ) -> Task<Result<Option<Vec<UpstreamProvider>>>> {
        let configured = Self::settings(cx)
            .available_models
            .iter()
            .any(|configured| configured.name == model.id().0.as_ref());
        if configured {
            return Task::ready(Ok(None));
        }

        let model_id = model.id().0.to_string();
        let now = cx.background_executor().now();
        let fresh = self
            .state
            .read(cx)
            .upstream_providers
            .get(&model_id)
            .filter(|cached| now.duration_since(cached.fetched_at) < PROVIDER_ENDPOINTS_TTL)
            .map(|cached| cached.providers.clone());
        if let Some(providers) = fresh {
            return Task::ready(Ok(Some(providers)));
        }

        let http_client = self.http_client.clone();
        let api_url = Self::api_url(cx);
        let extra_headers = Self::settings(cx).custom_headers.clone();
        let Some(api_key) = self.state.read(cx).api_key_state.key(&api_url) else {
            return Task::ready(Err(anyhow::anyhow!(
                "No API key is configured for {}",
                PROVIDER_NAME.0
            )));
        };

        let state = self.state.clone();
        cx.spawn(async move |cx| {
            let endpoints = match list_provider_endpoints(
                http_client.as_ref(),
                &api_url,
                &api_key,
                &model_id,
                &extra_headers,
            )
            .await
            {
                Ok(endpoints) => endpoints,
                Err(error) => {
                    let error = LanguageModelCompletionError::from(error);
                    let stale = state.read_with(cx, |state, _cx| {
                        state
                            .upstream_providers
                            .get(&model_id)
                            .map(|cached| cached.providers.clone())
                    });
                    let Some(providers) = stale else {
                        return Err(error.into());
                    };
                    log::warn!(
                        "Failed to refresh upstream providers for {model_id}, using the cached list: {error:#}"
                    );
                    return Ok(Some(providers));
                }
            };

            let providers = upstream_providers_from_endpoints(&endpoints);
            let fetched_at = cx.background_executor().now();
            state.update(cx, |state, cx| {
                state.upstream_providers.insert(
                    model_id,
                    CachedUpstreamProviders {
                        providers: providers.clone(),
                        fetched_at,
                    },
                );
                cx.notify();
            });
            Ok(Some(providers))
        })
    }

    fn upstream_provider(
        &self,
        model: &LanguageModel,
        provider: &str,
        cx: &App,
    ) -> Option<UpstreamProvider> {
        if Self::settings(cx)
            .available_models
            .iter()
            .any(|configured| configured.name == model.id().0.as_ref())
        {
            return None;
        }
        let cached = self
            .state
            .read(cx)
            .upstream_providers
            .get(model.id().0.as_ref())?;
        cached
            .providers
            .iter()
            .find(|listed| listed.id == provider)
            .cloned()
    }

    fn is_authenticated(&self, cx: &App) -> bool {
        self.state.read(cx).is_authenticated()
    }

    fn authenticate(&self, cx: &mut App) -> Task<Result<(), AuthenticateError>> {
        self.state.update(cx, |state, cx| state.authenticate(cx))
    }

    fn refresh_models(&self, cx: &mut App) {
        self.state
            .update(cx, |state, cx| state.restart_fetch_models_task(cx));
    }

    fn settings_view(&self, cx: &mut App) -> Option<ProviderSettingsView> {
        let state = self.state.read(cx);
        Some(ProviderSettingsView::ApiKey(ApiKeyConfiguration::new(
            state.api_key_state.has_key(),
            state.api_key_state.is_from_env_var(),
            state.api_key_state.env_var_name().clone(),
            "https://openrouter.ai/keys".into(),
        )))
    }

    fn set_api_key(&self, api_key: Option<String>, cx: &mut App) -> Task<Result<()>> {
        self.state
            .update(cx, |state, cx| state.set_api_key(api_key, cx))
    }
}

fn upstream_providers_from_endpoints(endpoints: &[ProviderEndpoint]) -> Vec<UpstreamProvider> {
    let mut providers: Vec<UpstreamProvider> = endpoints
        .iter()
        .filter_map(|endpoint| {
            let id = endpoint.slug.clone()?;
            let mut tags = Vec::new();
            // The slug variant stands in for the service tier when the API
            // reports none.
            let slug_variant = endpoint
                .slug
                .as_deref()
                .and_then(|slug| slug.split_once('/'))
                .map(|(_, variant)| variant);
            if let Some(service_tier) = endpoint.service_tier.as_deref().or(slug_variant) {
                tags.push(service_tier.to_string());
            }
            if let Some(quantization) = endpoint
                .quantization
                .as_deref()
                .filter(|quantization| *quantization != "unknown")
            {
                tags.push(quantization.to_string());
            }
            tags.dedup();
            let pricing = endpoint.pricing.as_ref().and_then(|pricing| {
                let input = pricing.prompt.map(|price| price * 1_000_000.0);
                let output = pricing.completion.map(|price| price * 1_000_000.0);
                let cache_read = pricing.input_cache_read.map(|price| price * 1_000_000.0);
                (input.is_some() || output.is_some() || cache_read.is_some()).then_some(
                    UpstreamProviderPricing {
                        input,
                        output,
                        cache_read,
                        discount: pricing.discount,
                    },
                )
            });
            Some(UpstreamProvider {
                id,
                name: endpoint.provider_name.clone(),
                tags,
                pricing,
                throughput: endpoint.throughput_last_30m.as_ref().map(|stats| stats.p50),
                latency: endpoint.latency_last_30m.as_ref().map(|stats| stats.p50),
                context_length: endpoint.context_length,
            })
        })
        .collect();

    // Fastest first. Endpoints without throughput data go last.
    providers.sort_by(|left, right| {
        right
            .throughput
            .partial_cmp(&left.throughput)
            .unwrap_or(Ordering::Equal)
    });
    providers
}

impl LanguageModelClient for OpenRouterLanguageModelProvider {
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
        let request_limiter = self.request_limiters.for_model(&model.id);
        let openrouter_request =
            match into_open_router(request, &config, config.max_output_tokens()) {
                Ok(request) => request,
                Err(error) => return async move { Err(error.into()) }.boxed(),
            };
        let request = self.stream_open_router_request(openrouter_request, cx);
        let executor = cx.background_executor().clone();
        let future = request_limiter.stream(async move {
            let response = request.await?;
            let events = ChatCompletionEventMapper::new().map_stream(response);
            Ok(language_model::stream_in_background(
                events.boxed(),
                executor,
            ))
        });
        async move { Ok(future.await?.boxed()) }.boxed()
    }
}

fn language_model(model: &open_router::Model) -> LanguageModel {
    let efforts: &[ReasoningEffort] = if !model.supported_efforts.is_empty() {
        &model.supported_efforts
    } else if model.supports_max_tokens {
        &ReasoningEffort::OPENAI_COMPATIBLE_SELECTABLE
    } else {
        &[]
    };
    let default_effort = model
        .default_effort
        .or_else(|| model.supports_max_tokens.then_some(ReasoningEffort::Medium));
    let supported_effort_levels = efforts
        .iter()
        .map(|&effort| LanguageModelEffortLevel {
            name: effort.label().into(),
            value: effort.value().into(),
            is_default: Some(effort) == default_effort,
            supports_disabling_thinking: None,
        })
        .collect();
    LanguageModel {
        supports_tools: model.supports_tool_calls(),
        supports_streaming_tools: true,
        supports_thinking: matches!(
            model.mode,
            OpenRouterModelMode::Thinking { .. } | OpenRouterModelMode::Adaptive
        ),
        supports_disabling_thinking: !model.mandatory_reasoning,
        supported_effort_levels,
        tool_choice_support: LanguageModelToolChoiceSupport::ALL,
        supports_images: model.supports_images.unwrap_or(false),
        ..LanguageModel::new(
            LanguageModelId::from(model.id().to_string()),
            LanguageModelName::from(model.display_name().to_string()),
            PROVIDER_ID,
            PROVIDER_NAME,
            format!("openrouter/{}", model.id()),
            model.max_token_count(),
        )
    }
}

pub fn into_open_router(
    request: LanguageModelRequest,
    model: &Model,
    max_output_tokens: Option<u64>,
) -> Result<open_router::Request> {
    let max_output_tokens = request.effective_max_output_tokens(max_output_tokens);
    if request.contains_custom_tool_input() {
        anyhow::bail!("OpenRouter does not support custom tools");
    }

    // Anthropic models via OpenRouter don't accept reasoning_details being echoed back
    // in requests - it's an output-only field for them. However, Gemini models require
    // the thought signatures to be echoed back for proper reasoning chain continuity.
    // Note: OpenRouter's model API provides an `architecture.tokenizer` field (e.g. "Claude",
    // "Gemini") which could replace this ID prefix check, but since this is the only place
    // we need this distinction, we're just using this less invasive check instead.
    // If we ever have a more formal distionction between the models in the future,
    // we should revise this to use that instead.
    let is_anthropic_model = model.id().starts_with("anthropic/");
    let session_id = open_router_session_id(request.thread_id);

    let mut messages = Vec::new();
    let mut any_message_wants_cache = false;
    let mut last_cache_message_index: Option<usize> = None;

    for message in request.messages {
        let mut message_added_content = false;
        let reasoning_details_for_message = if is_anthropic_model {
            None
        } else {
            message.reasoning_details.clone()
        };

        let message_wants_cache = message.cache;
        if message_wants_cache {
            any_message_wants_cache = true;
        }

        for content in message.content {
            match content {
                MessageContent::Text(text) => {
                    add_message_content_part(
                        open_router::MessagePart::Text {
                            text,
                            cache_control: None,
                        },
                        message.role,
                        &mut messages,
                        reasoning_details_for_message.clone(),
                    );
                    message_added_content = true;
                }
                MessageContent::Thinking { .. } => {}
                MessageContent::RedactedThinking(_) => {}
                MessageContent::Compaction(_) => {}
                MessageContent::Image(image) => {
                    add_message_content_part(
                        open_router::MessagePart::Image {
                            image_url: image.to_base64_url(),
                        },
                        message.role,
                        &mut messages,
                        reasoning_details_for_message.clone(),
                    );
                    message_added_content = true;
                }
                MessageContent::ToolUse(tool_use) => {
                    let input = tool_use.input.as_json().ok_or_else(|| {
                        anyhow::anyhow!("OpenRouter does not support custom tool calls")
                    })?;
                    let tool_call = open_router::ToolCall {
                        id: tool_use.id.to_string(),
                        content: open_router::ToolCallContent::Function {
                            function: open_router::FunctionContent {
                                name: tool_use.name.to_string(),
                                arguments: serde_json::to_string(input).unwrap_or_default(),
                                thought_signature: tool_use.thought_signature.clone(),
                            },
                        },
                    };

                    if let Some(open_router::RequestMessage::Assistant { tool_calls, .. }) =
                        messages.last_mut()
                    {
                        tool_calls.push(tool_call);
                    } else {
                        messages.push(open_router::RequestMessage::Assistant {
                            content: None,
                            tool_calls: vec![tool_call],
                            reasoning_details: reasoning_details_for_message.clone(),
                        });
                    }
                    message_added_content = true;
                }
                MessageContent::ToolResult(tool_result) => {
                    let content: Vec<open_router::MessagePart> = tool_result
                        .content
                        .iter()
                        .map(|part| match part {
                            LanguageModelToolResultContent::Text(text) => {
                                open_router::MessagePart::Text {
                                    text: text.to_string(),
                                    cache_control: None,
                                }
                            }
                            LanguageModelToolResultContent::Image(image) => {
                                open_router::MessagePart::Image {
                                    image_url: image.to_base64_url(),
                                }
                            }
                        })
                        .collect();

                    messages.push(open_router::RequestMessage::Tool {
                        content: content.into(),
                        tool_call_id: tool_result.tool_use_id.to_string(),
                    });
                    message_added_content = true;
                }
            }
        }

        if message_wants_cache && message_added_content {
            last_cache_message_index = messages.len().checked_sub(1);
        }
    }

    if is_anthropic_model && any_message_wants_cache {
        // OpenRouter's top-level automatic cache_control restricts routing to
        // Anthropic direct; explicit block breakpoints also work on Bedrock and Vertex.
        if let Some(content) = last_cache_message_index
            .and_then(|index| messages.get_mut(index))
            .and_then(request_message_content_mut)
        {
            set_last_text_cache_control(content, cache_control(None));
        }

        if let Some(content) = messages.iter_mut().find_map(|message| match message {
            open_router::RequestMessage::System { content } => Some(content),
            _ => None,
        }) {
            set_last_text_cache_control(
                content,
                cache_control(Some(open_router::CacheTtl::OneHour)),
            );
        }
    }

    Ok(open_router::Request {
        model: model.id().into(),
        messages,
        stream: true,
        session_id,
        stop: request.stop,
        temperature: request.temperature.unwrap_or(0.4),
        max_tokens: max_output_tokens,
        parallel_tool_calls: (!request.tools.is_empty() && !model.supports_parallel_tool_calls())
            .then_some(false),
        usage: open_router::RequestUsage { include: true },
        reasoning: match model.mode {
            OpenRouterModelMode::Adaptive if request.thinking_allowed => {
                Some(open_router::Reasoning {
                    enabled: Some(true),
                    effort: request
                        .thinking_effort
                        .as_deref()
                        .and_then(|e| e.parse::<ReasoningEffort>().ok()),
                    max_tokens: None,
                    exclude: None,
                })
            }
            OpenRouterModelMode::Thinking { budget_tokens } if request.thinking_allowed => {
                Some(open_router::Reasoning {
                    enabled: Some(true),
                    effort: None,
                    max_tokens: budget_tokens,
                    exclude: None,
                })
            }
            OpenRouterModelMode::Adaptive | OpenRouterModelMode::Thinking { .. }
                if !model.mandatory_reasoning =>
            {
                Some(open_router::Reasoning {
                    enabled: Some(false),
                    effort: None,
                    max_tokens: None,
                    exclude: None,
                })
            }
            _ => None,
        },
        tools: request
            .tools
            .into_iter()
            .map(|tool| {
                let input_schema = match tool.input {
                    language_model::LanguageModelRequestToolInput::Function {
                        input_schema,
                        ..
                    } => input_schema,
                    language_model::LanguageModelRequestToolInput::Custom { .. } => {
                        return Err(anyhow::anyhow!("OpenRouter does not support custom tools"));
                    }
                };
                Ok(open_router::ToolDefinition::Function {
                    function: open_router::FunctionDefinition {
                        name: tool.name,
                        description: Some(tool.description),
                        parameters: Some(input_schema),
                    },
                })
            })
            .collect::<Result<_>>()?,
        tool_choice: request.tool_choice.map(|choice| match choice {
            LanguageModelToolChoice::Auto => open_router::ToolChoice::Auto,
            LanguageModelToolChoice::Any => open_router::ToolChoice::Required,
            LanguageModelToolChoice::None => open_router::ToolChoice::None,
        }),
        provider: request
            .upstream_provider
            .map(|upstream_provider| {
                let mut provider = model.provider.clone().unwrap_or_default();
                provider.only = Some(vec![upstream_provider]);
                provider
            })
            .or_else(|| model.provider.clone()),
    })
}

fn open_router_session_id(thread_id: Option<String>) -> Option<String> {
    thread_id.map(|thread_id| {
        let mut hasher = Sha256::new();
        hasher.update(b"zed-openrouter-session-v1\0");
        hasher.update(thread_id.as_bytes());
        format!("{:x}", hasher.finalize())
    })
}

fn cache_control(ttl: Option<open_router::CacheTtl>) -> open_router::CacheControl {
    open_router::CacheControl {
        cache_type: open_router::CacheControlType::Ephemeral,
        ttl,
    }
}

fn request_message_content_mut(
    message: &mut open_router::RequestMessage,
) -> Option<&mut open_router::MessageContent> {
    match message {
        open_router::RequestMessage::User { content }
        | open_router::RequestMessage::System { content }
        | open_router::RequestMessage::Tool { content, .. } => Some(content),
        open_router::RequestMessage::Assistant {
            content: Some(content),
            ..
        } => Some(content),
        open_router::RequestMessage::Assistant { content: None, .. } => None,
    }
}

fn set_last_text_cache_control(
    content: &mut open_router::MessageContent,
    cache_control: open_router::CacheControl,
) {
    match content {
        open_router::MessageContent::Plain(text) => {
            let text = std::mem::take(text);
            *content =
                open_router::MessageContent::Multipart(vec![open_router::MessagePart::Text {
                    text,
                    cache_control: Some(cache_control),
                }]);
        }
        open_router::MessageContent::Multipart(parts) => {
            for part in parts.iter_mut().rev() {
                if let open_router::MessagePart::Text {
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

fn add_message_content_part(
    new_part: open_router::MessagePart,
    role: Role,
    messages: &mut Vec<open_router::RequestMessage>,
    reasoning_details: Option<Arc<serde_json::Value>>,
) {
    match (role, messages.last_mut()) {
        (Role::User, Some(open_router::RequestMessage::User { content }))
        | (Role::System, Some(open_router::RequestMessage::System { content })) => {
            content.push_part(new_part);
        }
        (
            Role::Assistant,
            Some(open_router::RequestMessage::Assistant {
                content: Some(content),
                ..
            }),
        ) => {
            content.push_part(new_part);
        }
        _ => {
            messages.push(match role {
                Role::User => open_router::RequestMessage::User {
                    content: open_router::MessageContent::from(vec![new_part]),
                },
                Role::Assistant => open_router::RequestMessage::Assistant {
                    content: Some(open_router::MessageContent::from(vec![new_part])),
                    tool_calls: Vec::new(),
                    reasoning_details,
                },
                Role::System => open_router::RequestMessage::System {
                    content: open_router::MessageContent::from(vec![new_part]),
                },
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpui::BorrowAppContext;
    use open_router::{PercentileStats, ProviderPricing};

    #[gpui::test]
    async fn test_session_id_is_stable_without_exposing_thread_id() {
        let model = open_router::Model::new(
            "openai/gpt-4o",
            Some("GPT-4o"),
            Some(128000),
            Some(true),
            Some(false),
            None,
            None,
            None,
            false,
            None,
        );
        let thread_id = "internal-thread-id";
        let request = |max_output_tokens| LanguageModelRequest {
            thread_id: Some(thread_id.to_string()),
            max_output_tokens,
            messages: vec![language_model::LanguageModelRequestMessage {
                role: Role::User,
                content: vec![MessageContent::Text("Hello".to_string())],
                cache: false,
                reasoning_details: None,
            }],
            ..Default::default()
        };

        for (requested, maximum, expected) in [
            (None, None, None),
            (None, Some(4096), Some(4096)),
            (Some(1024), Some(4096), Some(1024)),
            (Some(8192), Some(4096), Some(4096)),
        ] {
            let result = into_open_router(request(requested), &model, maximum).unwrap();
            assert_eq!(
                result.session_id,
                open_router_session_id(Some(thread_id.into()))
            );
            assert_ne!(result.session_id.as_deref(), Some(thread_id));
            assert_ne!(
                result.session_id,
                open_router_session_id(Some("another-thread-id".into()))
            );
            assert_eq!(
                serde_json::to_value(result).unwrap()["max_tokens"].as_u64(),
                expected
            );
        }
    }

    #[gpui::test]
    async fn test_anthropic_model_caching_two_tier() {
        let model = open_router::Model::new(
            "anthropic/claude-sonnet-4-5",
            Some("Claude Sonnet"),
            Some(200000),
            Some(true),
            Some(false),
            None,
            None,
            None,
            false,
            None,
        );

        let request = LanguageModelRequest {
            messages: vec![
                language_model::LanguageModelRequestMessage {
                    role: Role::System,
                    content: vec![MessageContent::Text("You are helpful.".to_string())],
                    cache: false,
                    reasoning_details: None,
                },
                language_model::LanguageModelRequestMessage {
                    role: Role::User,
                    content: vec![MessageContent::Text("Hello".to_string())],
                    cache: false,
                    reasoning_details: None,
                },
                language_model::LanguageModelRequestMessage {
                    role: Role::Assistant,
                    content: vec![MessageContent::Text("Hi there!".to_string())],
                    cache: false,
                    reasoning_details: None,
                },
                language_model::LanguageModelRequestMessage {
                    role: Role::User,
                    content: vec![MessageContent::Text("What is 2+2?".to_string())],
                    cache: true,
                    reasoning_details: None,
                },
            ],
            stop: vec![],
            temperature: None,
            tools: vec![],
            tool_choice: None,
            thinking_allowed: false,
            thinking_effort: None,
            speed: None,
            upstream_provider: None,
            thread_id: None,
            prompt_cache_key: None,
            prompt_id: None,
            intent: None,
            compact_at_tokens: None,
            max_output_tokens: None,
        };

        let result = into_open_router(request, &model, None).unwrap();

        let system_cache = result.messages.iter().find_map(|m| {
            if let open_router::RequestMessage::System { content } = m {
                if let open_router::MessageContent::Multipart(parts) = content {
                    parts.iter().last().and_then(|p| {
                        if let open_router::MessagePart::Text { cache_control, .. } = p {
                            *cache_control
                        } else {
                            None
                        }
                    })
                } else {
                    None
                }
            } else {
                None
            }
        });
        assert!(
            matches!(
                system_cache,
                Some(open_router::CacheControl {
                    cache_type: open_router::CacheControlType::Ephemeral,
                    ttl: Some(open_router::CacheTtl::OneHour),
                })
            ),
            "System message should have 1h cache_control, got: {system_cache:?}"
        );

        let tail_cache = result.messages.last().and_then(|last_message| {
            if let open_router::RequestMessage::User { content } = last_message {
                if let open_router::MessageContent::Multipart(parts) = content {
                    parts.iter().last().and_then(|part| {
                        if let open_router::MessagePart::Text { cache_control, .. } = part {
                            *cache_control
                        } else {
                            None
                        }
                    })
                } else {
                    None
                }
            } else {
                None
            }
        });
        assert!(
            matches!(
                tail_cache,
                Some(open_router::CacheControl {
                    cache_type: open_router::CacheControlType::Ephemeral,
                    ttl: None,
                })
            ),
            "Last cache:true message should have 5min cache_control, got: {tail_cache:?}"
        );

        for (i, message) in result.messages.iter().enumerate() {
            let is_system = matches!(message, open_router::RequestMessage::System { .. });
            let is_last = i == result.messages.len() - 1;
            if is_system || is_last {
                continue;
            }
            let parts: Option<&Vec<open_router::MessagePart>> = match message {
                open_router::RequestMessage::User { content }
                | open_router::RequestMessage::System { content }
                | open_router::RequestMessage::Tool { content, .. } => {
                    if let open_router::MessageContent::Multipart(parts) = content {
                        Some(parts)
                    } else {
                        None
                    }
                }
                open_router::RequestMessage::Assistant {
                    content: Some(content),
                    ..
                } => {
                    if let open_router::MessageContent::Multipart(parts) = content {
                        Some(parts)
                    } else {
                        None
                    }
                }
                _ => None,
            };
            if let Some(parts) = parts {
                for part in parts {
                    if let open_router::MessagePart::Text { cache_control, .. } = part {
                        assert!(
                            cache_control.is_none(),
                            "Message {i} should not have cache_control"
                        );
                    }
                }
            }
        }
    }

    #[gpui::test]
    async fn test_anthropic_model_no_cache_when_no_cache_flag() {
        let model = open_router::Model::new(
            "anthropic/claude-sonnet-4-5",
            Some("Claude Sonnet"),
            Some(200000),
            Some(true),
            Some(false),
            None,
            None,
            None,
            false,
            None,
        );

        let request = LanguageModelRequest {
            messages: vec![
                language_model::LanguageModelRequestMessage {
                    role: Role::System,
                    content: vec![MessageContent::Text("You are helpful.".to_string())],
                    cache: false,
                    reasoning_details: None,
                },
                language_model::LanguageModelRequestMessage {
                    role: Role::User,
                    content: vec![MessageContent::Text("Hello".to_string())],
                    cache: false,
                    reasoning_details: None,
                },
            ],
            stop: vec![],
            temperature: None,
            tools: vec![],
            tool_choice: None,
            thinking_allowed: false,
            thinking_effort: None,
            speed: None,
            upstream_provider: None,
            thread_id: None,
            prompt_cache_key: None,
            prompt_id: None,
            intent: None,
            compact_at_tokens: None,
            max_output_tokens: None,
        };

        let result = into_open_router(request, &model, None).unwrap();

        for message in &result.messages {
            let content = match message {
                open_router::RequestMessage::User { content }
                | open_router::RequestMessage::System { content } => Some(content),
                _ => None,
            };
            if let Some(content) = content {
                if let open_router::MessageContent::Multipart(parts) = content {
                    for part in parts {
                        if let open_router::MessagePart::Text { cache_control, .. } = part {
                            assert!(
                                cache_control.is_none(),
                                "No message should have cache_control when no cache:true flags"
                            );
                        }
                    }
                }
            }
        }
    }

    #[gpui::test]
    async fn test_non_anthropic_model_no_cache_control() {
        let model = open_router::Model::new(
            "openai/gpt-4o",
            Some("GPT-4o"),
            Some(128000),
            Some(true),
            Some(false),
            None,
            None,
            None,
            false,
            None,
        );

        let request = LanguageModelRequest {
            messages: vec![
                language_model::LanguageModelRequestMessage {
                    role: Role::System,
                    content: vec![MessageContent::Text("You are helpful.".to_string())],
                    cache: false,
                    reasoning_details: None,
                },
                language_model::LanguageModelRequestMessage {
                    role: Role::User,
                    content: vec![MessageContent::Text("Hello".to_string())],
                    cache: true,
                    reasoning_details: None,
                },
            ],
            stop: vec![],
            temperature: None,
            tools: vec![],
            tool_choice: None,
            thinking_allowed: false,
            thinking_effort: None,
            speed: None,
            upstream_provider: None,
            thread_id: None,
            prompt_cache_key: None,
            prompt_id: None,
            intent: None,
            compact_at_tokens: None,
            max_output_tokens: None,
        };

        let result = into_open_router(request, &model, None).unwrap();

        for message in &result.messages {
            let content = match message {
                open_router::RequestMessage::User { content }
                | open_router::RequestMessage::System { content } => Some(content),
                _ => None,
            };
            if let Some(content) = content {
                if let open_router::MessageContent::Multipart(parts) = content {
                    for part in parts {
                        if let open_router::MessagePart::Text { cache_control, .. } = part {
                            assert!(
                                cache_control.is_none(),
                                "Non-Anthropic model should never have cache_control"
                            );
                        }
                    }
                }
            }
        }
    }

    #[gpui::test]
    async fn test_into_open_router_sends_requested_effort() {
        let model = open_router::Model::new(
            "z-ai/glm-5.2",
            Some("GLM 5.2"),
            Some(1_048_576),
            Some(true),
            Some(false),
            Some(OpenRouterModelMode::Adaptive),
            Some(vec![
                open_router::ReasoningEffort::XHigh,
                open_router::ReasoningEffort::High,
            ]),
            Some(open_router::ReasoningEffort::High),
            false,
            None,
        );
        let request = LanguageModelRequest {
            messages: vec![language_model::LanguageModelRequestMessage {
                role: Role::User,
                content: vec![MessageContent::Text("Hello".to_string())],
                cache: false,
                reasoning_details: None,
            }],
            thinking_allowed: true,
            thinking_effort: Some("xhigh".to_string()),
            ..Default::default()
        };

        let result = into_open_router(request, &model, None).unwrap();
        let reasoning = result.reasoning.expect("reasoning should be set");
        assert_eq!(reasoning.effort, Some(open_router::ReasoningEffort::XHigh));
        assert_eq!(
            reasoning.max_tokens, None,
            "max_tokens should not be sent when effort is used"
        );
        assert_eq!(reasoning.exclude, None);
        assert_eq!(reasoning.enabled, Some(true));
    }

    #[gpui::test]
    async fn test_into_open_router_disables_reasoning_when_thinking_not_allowed() {
        let model = open_router::Model::new(
            "z-ai/glm-5.2",
            Some("GLM 5.2"),
            Some(1_048_576),
            Some(true),
            Some(false),
            Some(OpenRouterModelMode::Adaptive),
            Some(vec![
                open_router::ReasoningEffort::XHigh,
                open_router::ReasoningEffort::High,
            ]),
            Some(open_router::ReasoningEffort::High),
            false,
            None,
        );
        let request = LanguageModelRequest {
            messages: vec![language_model::LanguageModelRequestMessage {
                role: Role::User,
                content: vec![MessageContent::Text("Hello".to_string())],
                cache: false,
                reasoning_details: None,
            }],
            thinking_allowed: false,
            thinking_effort: Some("high".to_string()),
            ..Default::default()
        };

        let result = into_open_router(request, &model, None).unwrap();
        let reasoning = result.reasoning.expect("reasoning should be set");
        assert_eq!(reasoning.enabled, Some(false));
        assert_eq!(
            reasoning.effort, None,
            "effort should not be sent when reasoning is disabled"
        );
        assert_eq!(reasoning.max_tokens, None);
    }

    #[test]
    fn into_open_router_uses_requested_upstream_provider() {
        let model = open_router::Model::new(
            "openai/gpt-4",
            Some("GPT-4"),
            Some(8192),
            Some(true),
            Some(false),
            None,
            None,
            None,
            false,
            None,
        );
        let request = LanguageModelRequest {
            upstream_provider: Some("openai".to_string()),
            ..Default::default()
        };

        let result = into_open_router(request, &model, None).unwrap();
        let provider = result.provider.expect("provider should be set");
        assert_eq!(provider.only, Some(vec!["openai".to_string()]));
    }

    #[test]
    fn into_open_router_falls_back_to_model_provider() {
        let model = open_router::Model::new(
            "openai/gpt-4",
            Some("GPT-4"),
            Some(8192),
            Some(true),
            Some(false),
            None,
            None,
            None,
            false,
            Some(provider_with_only("azure")),
        );

        let result = into_open_router(LanguageModelRequest::default(), &model, None).unwrap();
        let provider = result.provider.expect("provider from model");
        assert_eq!(provider.only, Some(vec!["azure".to_string()]));
    }

    #[test]
    fn into_open_router_merges_upstream_provider_into_model_provider() {
        let model_provider: open_router::Provider = serde_json::from_value(serde_json::json!({
            "only": ["anthropic"],
            "order": ["anthropic"],
            "allow_fallbacks": false,
            "sort": "price"
        }))
        .unwrap();
        let model = open_router::Model::new(
            "openai/gpt-4",
            Some("GPT-4"),
            Some(8192),
            Some(true),
            Some(false),
            None,
            None,
            None,
            false,
            Some(model_provider),
        );
        let request = LanguageModelRequest {
            upstream_provider: Some("openai".to_string()),
            ..Default::default()
        };

        let result = into_open_router(request, &model, None).unwrap();
        let provider = result.provider.expect("provider should be set");
        let value = serde_json::to_value(&provider).unwrap();
        assert_eq!(value["only"], serde_json::json!(["openai"]));
        assert_eq!(value["order"], serde_json::json!(["anthropic"]));
        assert_eq!(value["allow_fallbacks"], serde_json::json!(false));
        assert_eq!(value["sort"], serde_json::json!("price"));
    }

    fn provider_with_only(only: &str) -> open_router::Provider {
        let mut provider = open_router::Provider::default();
        provider.only = Some(vec![only.to_string()]);
        provider
    }

    struct TestCredentialsProvider;

    impl CredentialsProvider for TestCredentialsProvider {
        fn read_credentials<'a>(
            &'a self,
            _url: &'a str,
            _cx: &'a AsyncApp,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = Result<Option<(String, Vec<u8>)>>> + 'a>,
        > {
            Box::pin(async { Ok(None) })
        }

        fn write_credentials<'a>(
            &'a self,
            _url: &'a str,
            _username: &'a str,
            _password: &'a [u8],
            _cx: &'a AsyncApp,
        ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<()>> + 'a>> {
            Box::pin(async { Ok(()) })
        }

        fn delete_credentials<'a>(
            &'a self,
            _url: &'a str,
            _cx: &'a AsyncApp,
        ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<()>> + 'a>> {
            Box::pin(async { Ok(()) })
        }
    }

    #[gpui::test]
    async fn restart_fetch_models_task_records_error_on_failure(cx: &mut gpui::TestAppContext) {
        cx.update(|cx| {
            let settings_store = SettingsStore::test(cx);
            cx.set_global(settings_store);
        });

        let http_client =
            http_client::FakeHttpClient::create(|_request| async move {
                Ok(http_client::Response::builder().status(401).body(
                    http_client::AsyncBody::from(r#"{"error":{"message":"invalid key"}}"#),
                )?)
            });
        let provider = cx.update(|cx| {
            OpenRouterLanguageModelProvider::new(http_client, Arc::new(TestCredentialsProvider), cx)
        });

        cx.update(|cx| provider.set_api_key(Some("test-key".to_string()), cx))
            .await
            .unwrap();
        cx.run_until_parked();

        let error = cx.read(|cx| provider.state.read(cx).last_fetch_models_error.clone());
        assert!(
            error.is_some(),
            "a failed model fetch should be recorded, not silently dropped"
        );
    }

    #[test]
    fn upstream_providers_from_endpoints_uses_slug_and_variant() {
        let endpoints = vec![
            ProviderEndpoint {
                name: "OpenAI: GPT-4".to_string(),
                provider_name: "OpenAI".to_string(),
                slug: Some("openai".to_string()),
                context_length: Some(8192),
                max_completion_tokens: Some(4096),
                quantization: Some("fp16".to_string()),
                service_tier: Some("flex".to_string()),
                status: Some(0),
                supported_parameters: vec!["tools".to_string()],
                pricing: Some(ProviderPricing {
                    prompt: Some(0.00003),
                    completion: Some(0.00006),
                    input_cache_read: Some(0.000015),
                    discount: Some(0.25),
                }),
                throughput_last_30m: Some(PercentileStats {
                    p50: 68.4,
                    p75: 55.1,
                    p90: 41.2,
                    p99: 22.9,
                }),
                latency_last_30m: Some(PercentileStats {
                    p50: 460.0,
                    p75: 720.0,
                    p90: 1100.0,
                    p99: 2400.0,
                }),
            },
            ProviderEndpoint {
                name: "DeepInfra (Turbo): GPT-4".to_string(),
                provider_name: "DeepInfra".to_string(),
                slug: Some("deepinfra/turbo".to_string()),
                context_length: Some(16384),
                max_completion_tokens: Some(8192),
                quantization: None,
                service_tier: None,
                status: Some(0),
                supported_parameters: Vec::new(),
                pricing: None,
                throughput_last_30m: None,
                latency_last_30m: None,
            },
            ProviderEndpoint {
                name: "No slug".to_string(),
                provider_name: "Unknown".to_string(),
                slug: None,
                context_length: None,
                max_completion_tokens: None,
                quantization: None,
                service_tier: None,
                status: None,
                supported_parameters: Vec::new(),
                pricing: None,
                throughput_last_30m: None,
                latency_last_30m: None,
            },
            ProviderEndpoint {
                name: "Together: GPT-4".to_string(),
                provider_name: "Together".to_string(),
                slug: Some("together".to_string()),
                context_length: Some(8192),
                max_completion_tokens: Some(4096),
                quantization: Some("unknown".to_string()),
                service_tier: Some("priority".to_string()),
                status: Some(0),
                supported_parameters: Vec::new(),
                pricing: None,
                throughput_last_30m: Some(PercentileStats {
                    p50: 120.0,
                    p75: 100.0,
                    p90: 80.0,
                    p99: 40.0,
                }),
                latency_last_30m: None,
            },
            ProviderEndpoint {
                name: "GMICloud: GPT-4".to_string(),
                provider_name: "GMICloud".to_string(),
                slug: Some("gmicloud/fp8".to_string()),
                context_length: Some(32768),
                max_completion_tokens: None,
                quantization: Some("fp8".to_string()),
                service_tier: None,
                status: Some(0),
                supported_parameters: Vec::new(),
                pricing: None,
                throughput_last_30m: None,
                latency_last_30m: None,
            },
        ];

        let options = upstream_providers_from_endpoints(&endpoints);

        assert_eq!(
            options,
            vec![
                UpstreamProvider {
                    id: "together".to_string(),
                    name: "Together".to_string(),
                    tags: vec!["priority".to_string()],
                    pricing: None,
                    throughput: Some(120.0),
                    latency: None,
                    context_length: Some(8192),
                },
                UpstreamProvider {
                    id: "openai".to_string(),
                    name: "OpenAI".to_string(),
                    tags: vec!["flex".to_string(), "fp16".to_string()],
                    pricing: Some(UpstreamProviderPricing {
                        input: Some(30.0),
                        output: Some(60.0),
                        cache_read: Some(15.0),
                        discount: Some(0.25),
                    }),
                    throughput: Some(68.4),
                    latency: Some(460.0),
                    context_length: Some(8192),
                },
                UpstreamProvider {
                    id: "deepinfra/turbo".to_string(),
                    name: "DeepInfra".to_string(),
                    tags: vec!["turbo".to_string()],
                    pricing: None,
                    throughput: None,
                    latency: None,
                    context_length: Some(16384),
                },
                UpstreamProvider {
                    id: "gmicloud/fp8".to_string(),
                    name: "GMICloud".to_string(),
                    tags: vec!["fp8".to_string()],
                    pricing: None,
                    throughput: None,
                    latency: None,
                    context_length: Some(32768),
                },
            ]
        );
    }

    #[gpui::test]
    async fn upstream_provider_reports_endpoint_limits(cx: &mut gpui::TestAppContext) {
        let provider = provider_with_http(cx, |request| {
            if request.uri().path().ends_with("/endpoints") {
                http_client::Response::builder()
                    .status(200)
                    .body(http_client::AsyncBody::from(ENDPOINTS_RESPONSE))
                    .expect("endpoints response")
            } else {
                http_client::Response::builder()
                    .status(200)
                    .body(http_client::AsyncBody::from(MODELS_RESPONSE))
                    .expect("models response")
            }
        });
        let model = authenticated_model(&provider, cx).await;

        let unknown = cx.update(|cx| provider.upstream_provider(&model, "openai", cx));
        assert_eq!(
            unknown, None,
            "limits are unknown before the list is fetched"
        );

        let listed = cx
            .update(|cx| provider.upstream_providers(&model, cx))
            .await
            .expect("upstream providers")
            .expect("providers are listed")
            .into_iter()
            .find(|provider| provider.id == "openai")
            .expect("openai is listed");
        let single = cx
            .update(|cx| provider.upstream_provider(&model, "openai", cx))
            .expect("the provider is known after the list is fetched");

        assert_eq!(single, listed);
        assert_eq!(single.context_length, Some(8192));
    }

    #[gpui::test]
    async fn upstream_providers_are_none_for_configured_models(cx: &mut gpui::TestAppContext) {
        let provider = cx.update(|cx| {
            let settings_store = SettingsStore::test(cx);
            cx.set_global(settings_store);

            let content = serde_json::json!({
                "language_models": {
                    "open_router": {
                        "available_models": [{
                            "name": "openai/gpt-4",
                            "max_tokens": 8192,
                            "provider": {
                                "only": ["openai"]
                            }
                        }]
                    }
                }
            })
            .to_string();
            cx.update_global::<SettingsStore, _>(|store, cx| {
                store
                    .set_user_settings(&content, cx)
                    .expect("failed to parse test settings");
            });

            let http_client = http_client::FakeHttpClient::create(|_request| async move {
                unreachable!("upstream providers must not be fetched when only is configured")
            });
            OpenRouterLanguageModelProvider::new(http_client, Arc::new(TestCredentialsProvider), cx)
        });

        let model = cx.update(|cx| {
            provider
                .provided_models(cx)
                .into_iter()
                .find(|model| model.id().0.as_ref() == "openai/gpt-4")
                .expect("model from settings")
        });

        let options = cx
            .update(|cx| provider.upstream_providers(&model, cx))
            .await
            .expect("upstream providers");

        assert!(options.is_none());
    }

    const MODELS_RESPONSE: &str = r#"{
        "data": [
            {
                "id": "openai/gpt-4",
                "name": "OpenAI: GPT-4",
                "created": 0,
                "description": "",
                "context_length": 8192,
                "supported_parameters": ["tools"]
            }
        ]
    }"#;

    const ENDPOINTS_RESPONSE: &str = r#"{
        "data": {
            "id": "openai/gpt-4",
            "name": "GPT-4",
            "endpoints": [
                {
                    "name": "OpenAI: GPT-4",
                    "provider_name": "OpenAI",
                    "tag": "openai",
                    "context_length": 8192,
                    "status": 0,
                    "supported_parameters": ["tools"]
                }
            ]
        }
    }"#;

    /// Builds a provider whose HTTP client answers model and endpoint requests
    /// through `handler`.
    fn provider_with_http(
        cx: &mut gpui::TestAppContext,
        handler: impl Fn(
            http_client::Request<http_client::AsyncBody>,
        ) -> http_client::Response<http_client::AsyncBody>
        + Send
        + Sync
        + 'static,
    ) -> OpenRouterLanguageModelProvider {
        cx.update(|cx| {
            let settings_store = SettingsStore::test(cx);
            cx.set_global(settings_store);
            let http_client = http_client::FakeHttpClient::create(move |request| {
                let response = handler(request);
                async move { Ok(response) }
            });
            OpenRouterLanguageModelProvider::new(http_client, Arc::new(TestCredentialsProvider), cx)
        })
    }

    async fn authenticated_model(
        provider: &OpenRouterLanguageModelProvider,
        cx: &mut gpui::TestAppContext,
    ) -> LanguageModel {
        cx.update(|cx| provider.set_api_key(Some("test-key".to_string()), cx))
            .await
            .expect("api key");
        cx.run_until_parked();

        cx.update(|cx| {
            provider
                .provided_models(cx)
                .into_iter()
                .find(|model| model.id().0.as_ref() == "openai/gpt-4")
                .expect("model from the fetched list")
        })
    }

    #[gpui::test]
    async fn upstream_providers_are_cached_for_an_hour(cx: &mut gpui::TestAppContext) {
        let endpoint_requests = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let fail_endpoints = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let counter = endpoint_requests.clone();
        let failing = fail_endpoints.clone();
        let provider = provider_with_http(cx, move |request| {
            if !request.uri().path().ends_with("/endpoints") {
                return http_client::Response::builder()
                    .status(200)
                    .body(http_client::AsyncBody::from(MODELS_RESPONSE))
                    .expect("models response");
            }

            counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if failing.load(std::sync::atomic::Ordering::SeqCst) {
                return http_client::Response::builder()
                    .status(500)
                    .body(http_client::AsyncBody::from(
                        r#"{"error":{"code":500,"message":"boom"}}"#,
                    ))
                    .expect("error response");
            }
            http_client::Response::builder()
                .status(200)
                .body(http_client::AsyncBody::from(ENDPOINTS_RESPONSE))
                .expect("endpoints response")
        });

        let model = authenticated_model(&provider, cx).await;
        let requests = || endpoint_requests.load(std::sync::atomic::Ordering::SeqCst);

        let first = cx
            .update(|cx| provider.upstream_providers(&model, cx))
            .await
            .expect("upstream providers")
            .expect("providers are listed");
        assert_eq!(requests(), 1);

        cx.update(|cx| provider.upstream_providers(&model, cx))
            .await
            .expect("upstream providers");
        assert_eq!(requests(), 1, "a read inside the TTL uses the cache");

        cx.executor()
            .advance_clock(PROVIDER_ENDPOINTS_TTL + Duration::from_secs(1));
        cx.update(|cx| provider.upstream_providers(&model, cx))
            .await
            .expect("upstream providers");
        assert_eq!(requests(), 2, "a read after the TTL fetches again");

        cx.executor()
            .advance_clock(PROVIDER_ENDPOINTS_TTL + Duration::from_secs(1));
        fail_endpoints.store(true, std::sync::atomic::Ordering::SeqCst);
        let stale = cx
            .update(|cx| provider.upstream_providers(&model, cx))
            .await
            .expect("a failed refresh must not hide the cached list")
            .expect("providers are still listed");
        assert_eq!(stale, first);
    }
}
