use std::collections::BTreeMap;
use std::sync::Arc;

use agent_client_protocol::v2::schema as acp_v2;
use anyhow::Result;
use gpui::{App, SharedString, Task};
use heck::ToSnakeCase as _;
use language_model::{LanguageModelProviderId, LanguageModelToolResultContent};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use util::markdown::MarkdownInlineCode;

use crate::{
    AgentToolOutput, AnyAgentTool, MAX_TOOL_NAME_LENGTH, ToolCallEventStream, ToolInput,
    provider_compatible_tool_name,
};

const DEFERRED_TOOL_MARKER: &str = "(deferred)";
const SERVER_SEPARATOR: &str = "__";

/// Returns the name that the model sees for a context server tool. The name
/// starts with the server ID, so that the model knows the source of the tool.
pub(crate) fn context_server_tool_name(server_id: &str, tool_name: &str) -> String {
    let tool_name = provider_compatible_tool_name(tool_name);
    let available = MAX_TOOL_NAME_LENGTH.saturating_sub(tool_name.len() + SERVER_SEPARATOR.len());
    if available == 0 {
        return tool_name;
    }
    let mut server_name = provider_compatible_tool_name(&server_id.to_snake_case());
    server_name.truncate(available);
    format!("{server_name}{SERVER_SEPARATOR}{tool_name}")
}

/// Shows a tool to the model with only its name and the first sentence of its
/// description. The model must load the full description and input schema with
/// [`ToolSearchTool`] before it calls the tool.
pub struct DeferredTool {
    tool: Arc<dyn AnyAgentTool>,
}

impl DeferredTool {
    // Tools are stored as `Arc<dyn AnyAgentTool>`, which is not `Send`.
    #[allow(clippy::arc_with_non_send_sync)]
    pub fn wrap(tool: Arc<dyn AnyAgentTool>) -> Arc<dyn AnyAgentTool> {
        Arc::new(Self { tool })
    }
}

impl AnyAgentTool for DeferredTool {
    fn name(&self) -> SharedString {
        self.tool.name()
    }

    fn description(&self) -> SharedString {
        let description = self.tool.description();
        let summary = first_sentence(&description);
        if summary.is_empty() {
            DEFERRED_TOOL_MARKER.into()
        } else {
            format!("{summary} {DEFERRED_TOOL_MARKER}").into()
        }
    }

    fn kind(&self) -> acp_v2::ToolKind {
        self.tool.kind()
    }

    fn initial_title(&self, input: serde_json::Value, cx: &mut App) -> SharedString {
        self.tool.initial_title(input, cx)
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {},
            "additionalProperties": true,
        })
    }

    fn supports_input_streaming(&self) -> bool {
        self.tool.supports_input_streaming()
    }

    fn supports_provider(&self, provider: &LanguageModelProviderId) -> bool {
        self.tool.supports_provider(provider)
    }

    fn allow_in_restricted_mode(&self) -> bool {
        self.tool.allow_in_restricted_mode()
    }

    fn deferred_tool(&self) -> Option<&Arc<dyn AnyAgentTool>> {
        Some(&self.tool)
    }

    fn run(
        self: Arc<Self>,
        input: ToolInput<serde_json::Value>,
        event_stream: ToolCallEventStream,
        cx: &mut App,
    ) -> Task<Result<AgentToolOutput, AgentToolOutput>> {
        self.tool.clone().run(input, event_stream, cx)
    }

    fn replay(
        &self,
        input: serde_json::Value,
        output: serde_json::Value,
        event_stream: ToolCallEventStream,
        cx: &mut App,
    ) -> Result<()> {
        self.tool.replay(input, output, event_stream, cx)
    }
}

fn first_sentence(text: &str) -> &str {
    let text = text.trim_start();
    let line = text.lines().next().unwrap_or_default();
    let end = line
        .char_indices()
        .find(|(index, character)| {
            matches!(character, '.' | '!' | '?')
                && line[index + character.len_utf8()..]
                    .chars()
                    .next()
                    .is_none_or(char::is_whitespace)
        })
        .map_or(line.len(), |(index, character)| {
            index + character.len_utf8()
        });
    line[..end].trim()
}

/// Loads the full descriptions and input schemas of deferred tools.
///
/// Some tools are deferred: they are available, but their definitions show
/// only a summary that ends with "(deferred)" and a placeholder input schema.
/// Before you call a deferred tool, call this tool with its name to get its
/// full description and input schema. You can load more than one tool in one
/// call.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct ToolSearchToolInput {
    /// The names of the deferred tools to load.
    pub names: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ToolSearchToolOutput {
    loaded: Vec<serde_json::Value>,
    not_found: Vec<String>,
}

/// Gives the model the full definitions of the deferred tools of one request.
pub struct ToolSearchTool {
    deferred_tools: BTreeMap<SharedString, Arc<dyn AnyAgentTool>>,
}

impl ToolSearchTool {
    pub const NAME: &'static str = "tool_search";

    /// Adds a [`ToolSearchTool`] to `tools` if `tools` contains a [`DeferredTool`].
    pub fn add_if_needed(tools: &mut BTreeMap<SharedString, Arc<dyn AnyAgentTool>>) {
        let deferred_tools = tools
            .iter()
            .filter_map(|(name, tool)| Some((name.clone(), tool.deferred_tool()?.clone())))
            .collect::<BTreeMap<_, _>>();
        if deferred_tools.is_empty() {
            return;
        }
        #[allow(clippy::arc_with_non_send_sync)]
        let tool_search: Arc<dyn AnyAgentTool> = Arc::new(Self { deferred_tools });
        if let Some(shadowed) = tools.insert(Self::NAME.into(), tool_search) {
            log::warn!(
                "tool `{}` is not available because `{}` replaces it",
                shadowed.name(),
                Self::NAME
            );
        }
    }

    fn load(&self, input: ToolSearchToolInput) -> Result<AgentToolOutput, AgentToolOutput> {
        let mut definitions = Vec::new();
        let mut output = ToolSearchToolOutput {
            loaded: Vec::new(),
            not_found: Vec::new(),
        };
        for name in input.names {
            match self.deferred_tools.get(name.as_str()) {
                Some(tool) => {
                    let definition = serde_json::json!({
                        "name": name,
                        "description": tool.description(),
                        "parameters": tool.input_schema(),
                    });
                    definitions.push(format!("<function>{definition}</function>"));
                    output.loaded.push(definition);
                }
                None => output.not_found.push(name),
            }
        }

        let mut text = definitions.join("\n");
        if !output.not_found.is_empty() {
            if !text.is_empty() {
                text.push_str("\n\n");
            }
            text.push_str(&format!(
                "No deferred tools with these names exist: {}",
                output.not_found.join(", ")
            ));
        }

        let is_error = output.loaded.is_empty();
        let output = AgentToolOutput {
            llm_output: vec![LanguageModelToolResultContent::Text(text.into())],
            raw_output: serde_json::to_value(&output).unwrap_or_else(|error| {
                log::error!("Failed to serialize tool output: {error}");
                serde_json::Value::Null
            }),
        };
        if is_error { Err(output) } else { Ok(output) }
    }
}

impl AnyAgentTool for ToolSearchTool {
    fn name(&self) -> SharedString {
        Self::NAME.into()
    }

    fn description(&self) -> SharedString {
        let schema = schemars::schema_for!(ToolSearchToolInput);
        schema
            .get("description")
            .and_then(|description| description.as_str())
            .unwrap_or_default()
            .to_string()
            .into()
    }

    fn kind(&self) -> acp_v2::ToolKind {
        acp_v2::ToolKind::Search
    }

    fn initial_title(&self, input: serde_json::Value, _cx: &mut App) -> SharedString {
        match serde_json::from_value::<ToolSearchToolInput>(input) {
            Ok(input) if !input.names.is_empty() => {
                let names = input
                    .names
                    .iter()
                    .map(|name| {
                        let name = self
                            .deferred_tools
                            .get(name.as_str())
                            .map_or_else(|| name.clone(), |tool| tool.name().to_string());
                        MarkdownInlineCode(&name).to_string()
                    })
                    .collect::<Vec<_>>();
                format!("Load tools {}", names.join(", ")).into()
            }
            _ => "Load tools".into(),
        }
    }

    fn input_schema(&self) -> serde_json::Value {
        let mut schema =
            language_model::tool_schema::root_schema_for::<ToolSearchToolInput>().to_value();
        language_model::tool_schema::normalize_tool_schema(&mut schema);
        schema
    }

    fn run(
        self: Arc<Self>,
        input: ToolInput<serde_json::Value>,
        _event_stream: ToolCallEventStream,
        cx: &mut App,
    ) -> Task<Result<AgentToolOutput, AgentToolOutput>> {
        cx.spawn(async move |_cx| {
            let input = input.recv().await.map_err(AgentToolOutput::from)?;
            let input = serde_json::from_value::<ToolSearchToolInput>(input)
                .map_err(|error| AgentToolOutput::from(anyhow::Error::from(error)))?;
            self.load(input)
        })
    }

    fn replay(
        &self,
        _input: serde_json::Value,
        _output: serde_json::Value,
        _event_stream: ToolCallEventStream,
        _cx: &mut App,
    ) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::first_sentence;

    #[test]
    fn test_first_sentence() {
        assert_eq!(
            first_sentence("Gets a user. Returns details."),
            "Gets a user."
        );
        assert_eq!(
            first_sentence("  Gets a user\nReturns details."),
            "Gets a user"
        );
        assert_eq!(
            first_sentence("Reads v1.2 of a file! More."),
            "Reads v1.2 of a file!"
        );
        assert_eq!(first_sentence("Ends here."), "Ends here.");
        assert_eq!(first_sentence(""), "");
    }
}
