use std::fmt;

use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tracing::{debug, trace};

use crate::{
	llms::{self, LlmProvider, LlmResponseStream, LlmsError},
	utils::{
		default_parameters,
		sse::{SseError, SseResponse},
	},
};

const ANTHROPIC_VERSION: &str = "2023-06-01";
const DEFAULT_MAX_TOKENS: u32 = 8096;
/// Used when adaptive thinking is on; the model needs room for both
/// reasoning and the final answer within this single cap.
const EFFORT_MAX_TOKENS: u32 = 32768;

#[derive(Clone)]
pub struct Anthropic {
	pub client: Client,
	pub api_key: String,
}

impl Anthropic {
	pub fn new(api_key: String) -> Self {
		Self {
			client: Client::new(),
			api_key,
		}
	}

	pub async fn request(
		&self,
		req: &Request,
	) -> Result<ResponseStream, AnthropicError> {
		#[derive(Debug, Serialize)]
		struct ApiReq<'a> {
			model: &'a str,
			max_tokens: u32,
			#[serde(skip_serializing_if = "Option::is_none")]
			system: Option<&'a str>,
			messages: &'a Vec<ApiMessage>,
			#[serde(skip_serializing_if = "Vec::is_empty")]
			tools: &'a Vec<ApiTool>,
			#[serde(skip_serializing_if = "Option::is_none")]
			thinking: Option<Thinking>,
			#[serde(skip_serializing_if = "Option::is_none")]
			output_config: Option<OutputConfig>,
			stream: bool,
		}

		let (thinking, output_config) = match req.effort {
			Some(effort) => {
				(Some(Thinking::Adaptive), Some(OutputConfig { effort }))
			}
			None => (None, None),
		};

		let api_req = ApiReq {
			model: req.model.as_str(),
			max_tokens: req.max_tokens,
			system: req.system.as_deref(),
			messages: &req.messages,
			tools: &req.tools,
			thinking,
			output_config,
			stream: true,
		};

		trace!("{:?}", serde_json::to_string(&api_req));

		let resp = self
			.client
			.post("https://api.anthropic.com/v1/messages")
			.header("x-api-key", &self.api_key)
			.header("anthropic-version", ANTHROPIC_VERSION)
			.json(&api_req)
			.send()
			.await?;

		if !resp.status().is_success() {
			let status = resp.status();
			let body = resp.text().await?;
			return Err(AnthropicError::ResponseError { status, body });
		}

		Ok(ResponseStream::new(SseResponse::new(resp)))
	}
}

impl fmt::Debug for Anthropic {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		f.debug_struct("Anthropic")
			.field("api_key", &"***")
			.finish()
	}
}

impl LlmProvider for Anthropic {
	type Stream = ResponseStream;

	async fn request(
		&self,
		req: &llms::Request,
	) -> Result<Self::Stream, LlmsError> {
		let model = match &req.model {
			llms::Model::ClaudeFable5 => AnthropicModel::Fable5,
			llms::Model::ClaudeOpus4_8 => AnthropicModel::Opus4_8,
			llms::Model::ClaudeSonnet5 => AnthropicModel::Sonnet5,
			llms::Model::ClaudeHaiku4_5 => AnthropicModel::Haiku4_5,
			m => unreachable!("unsupported model: {m:?}"),
		};

		let system = if req.instructions.is_empty() {
			None
		} else {
			Some(req.instructions.clone())
		};

		// Haiku 4.5 doesn't support adaptive thinking; silently ignore.
		let effort = match (model, req.reasoning_effort) {
			(AnthropicModel::Haiku4_5, Some(_)) => {
				debug!("reasoning_effort is ignored for Claude Haiku 4.5");
				None
			}
			(_, None) => None,
			(_, Some(llms::ReasoningEffort::Low)) => Some(Effort::Low),
			(_, Some(llms::ReasoningEffort::Medium)) => Some(Effort::Medium),
			(_, Some(llms::ReasoningEffort::High)) => Some(Effort::High),
		};

		// Thinking tokens count toward max_tokens; give the model headroom
		// for both reasoning and the final answer when effort is set.
		let max_tokens = if effort.is_some() {
			EFFORT_MAX_TOKENS
		} else {
			DEFAULT_MAX_TOKENS
		};

		self.request(&Request {
			messages: req.input.iter().cloned().map(Into::into).collect(),
			model,
			system,
			tools: req.tools.iter().cloned().map(Into::into).collect(),
			max_tokens,
			effort,
		})
		.await
		.map_err(Into::into)
	}
}

#[derive(Debug)]
pub struct Request {
	pub messages: Vec<ApiMessage>,
	pub model: AnthropicModel,
	pub system: Option<String>,
	pub tools: Vec<ApiTool>,
	/// Max tokens for the whole response. Adaptive thinking tokens count
	/// toward this cap, so callers should pass a larger value when `effort`
	/// is set.
	pub max_tokens: u32,
	/// `output_config.effort`. `None` omits the thinking/output_config fields.
	pub effort: Option<Effort>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Thinking {
	Adaptive,
}

#[derive(Debug, Serialize)]
struct OutputConfig {
	effort: Effort,
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Effort {
	Low,
	Medium,
	High,
}

#[derive(Debug, Serialize)]
pub struct ApiMessage {
	pub role: ApiRole,
	pub content: ApiMessageContent,
}

impl From<llms::Input> for ApiMessage {
	fn from(input: llms::Input) -> Self {
		match input {
			llms::Input::Text { role, content } => ApiMessage {
				role: role.into(),
				content: ApiMessageContent::Text(content),
			},
			llms::Input::ToolCall {
				id, name, input, ..
			} => ApiMessage {
				role: ApiRole::Assistant,
				content: ApiMessageContent::Blocks(vec![
					ApiContentBlock::ToolUse { id, name, input },
				]),
			},
			llms::Input::ToolCallOutput { id, output } => ApiMessage {
				role: ApiRole::User,
				content: ApiMessageContent::Blocks(vec![
					ApiContentBlock::ToolResult {
						tool_use_id: id,
						content: output,
					},
				]),
			},
		}
	}
}

#[derive(Debug, Serialize, Clone, Copy)]
#[serde(rename_all = "lowercase")]
pub enum ApiRole {
	User,
	Assistant,
}

impl From<llms::Role> for ApiRole {
	fn from(role: llms::Role) -> Self {
		match role {
			llms::Role::User => ApiRole::User,
			llms::Role::Assistant => ApiRole::Assistant,
		}
	}
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum ApiMessageContent {
	Text(String),
	Blocks(Vec<ApiContentBlock>),
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ApiContentBlock {
	Text {
		text: String,
	},
	ToolUse {
		id: String,
		name: String,
		input: Value,
	},
	ToolResult {
		tool_use_id: String,
		content: String,
	},
}

#[derive(Debug, Serialize)]
pub struct ApiTool {
	pub name: String,
	pub description: String,
	/// Full JSON Schema object sent verbatim as `input_schema`.
	/// Anthropic requires at minimum `{ "type": "object", "properties": {} }`.
	pub input_schema: Value,
}

impl From<llms::Tool> for ApiTool {
	fn from(tool: llms::Tool) -> Self {
		ApiTool {
			name: tool.name,
			description: tool.description,
			input_schema: tool.parameters.unwrap_or_else(default_parameters),
		}
	}
}

#[derive(Debug, Clone, Copy)]
pub enum AnthropicModel {
	Fable5,
	Opus4_8,
	Sonnet5,
	Haiku4_5,
}

impl AnthropicModel {
	pub fn as_str(&self) -> &'static str {
		match self {
			AnthropicModel::Fable5 => "claude-fable-5",
			AnthropicModel::Opus4_8 => "claude-opus-4-8",
			AnthropicModel::Sonnet5 => "claude-sonnet-5",
			AnthropicModel::Haiku4_5 => "claude-haiku-4-5",
		}
	}
}

#[derive(Debug, Deserialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Event {
	MessageStart {
		message: MessageStartData,
	},
	ContentBlockStart {
		index: u32,
		content_block: ContentBlockStartData,
	},
	ContentBlockDelta {
		index: u32,
		delta: ContentDelta,
	},
	ContentBlockStop {
		index: u32,
	},
	MessageDelta {
		delta: MessageDeltaData,
		usage: Option<MessageDeltaUsage>,
	},
	MessageStop,
	Ping,
	Error {
		error: ApiError,
	},
}

#[derive(Debug, Deserialize, Clone)]
pub struct MessageStartData {
	pub id: String,
	pub model: String,
	pub usage: Option<MessageStartUsage>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct MessageStartUsage {
	pub input_tokens: u32,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlockStartData {
	Text {
		text: String,
	},
	ToolUse {
		id: String,
		name: String,
	},
	/// Adaptive-thinking content block. We don't surface reasoning text to
	/// callers, so the fields are deserialized but ignored.
	Thinking {
		#[serde(default)]
		thinking: String,
		#[serde(default)]
		signature: String,
	},
}

#[derive(Debug, Deserialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentDelta {
	TextDelta {
		text: String,
	},
	InputJsonDelta {
		partial_json: String,
	},
	/// Streaming delta for a `thinking` content block. Ignored.
	ThinkingDelta {
		thinking: String,
	},
	/// Streaming signature for a `thinking` block. Ignored.
	SignatureDelta {
		signature: String,
	},
}

#[derive(Debug, Deserialize, Clone)]
pub struct MessageDeltaData {
	pub stop_reason: Option<String>,
	pub stop_sequence: Option<String>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct MessageDeltaUsage {
	pub output_tokens: u32,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ApiError {
	#[serde(rename = "type")]
	pub error_type: String,
	pub message: String,
}

#[derive(Debug, thiserror::Error)]
pub enum AnthropicError {
	#[error("Invalid LLM response: {0}")]
	InvalidLlmResponse(String),
	#[error("Response error: status {status}, body {body}")]
	ResponseError { status: StatusCode, body: String },
	#[error("API error: {error_type}: {message}")]
	ApiError { error_type: String, message: String },
	#[error("Reqwest error: {0}")]
	ReqwestError(#[from] reqwest::Error),
}

impl From<AnthropicError> for LlmsError {
	fn from(e: AnthropicError) -> Self {
		match e {
			AnthropicError::InvalidLlmResponse(msg) => LlmsError::Response {
				status: StatusCode::OK,
				body: msg,
			},
			AnthropicError::ResponseError { status, body } => {
				LlmsError::Response { status, body }
			}
			AnthropicError::ApiError {
				error_type,
				message,
			} => LlmsError::Response {
				status: StatusCode::OK,
				body: format!("{error_type}: {message}"),
			},
			AnthropicError::ReqwestError(e) => LlmsError::Reqwest(e),
		}
	}
}

enum BlockAccumulator {
	Text {
		text: String,
	},
	ToolUse {
		id: String,
		name: String,
		input_json: String,
	},
	/// Thinking blocks are accumulated as a placeholder so block indices
	/// stay aligned with what Anthropic emits, but the content is dropped
	/// when building the final response.
	Thinking,
}

pub struct ResponseStream {
	inner: SseResponse,
	/// Content blocks accumulated in arrival order (Anthropic always sends
	/// them sequentially, so index == position in this Vec).
	blocks: Vec<BlockAccumulator>,
	/// Token usage accumulated across `message_start` (input) and
	/// `message_delta` (output) events. `None` until the first event with a
	/// `usage` payload arrives.
	usage: Option<llms::Usage>,
	done: bool,
}

impl std::fmt::Debug for ResponseStream {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("ResponseStream")
			.field("done", &self.done)
			.finish()
	}
}

impl ResponseStream {
	fn new(inner: SseResponse) -> Self {
		Self {
			inner,
			blocks: Vec::new(),
			usage: None,
			done: false,
		}
	}

	async fn next_event(&mut self) -> Option<Result<Event, SseError>> {
		match self.inner.next().await {
			Some(Ok(ev)) => {
				trace!("anthropic event: {ev:?}");
				Some(Ok(ev))
			}
			other => other,
		}
	}

	fn build_response(&mut self) -> Result<llms::Response, AnthropicError> {
		let mut output = Vec::new();
		for block in self.blocks.drain(..) {
			match block {
				BlockAccumulator::Text { text } if !text.is_empty() => {
					output.push(llms::Output::Text { content: text });
				}
				BlockAccumulator::Text { .. } => {}
				BlockAccumulator::Thinking => {}
				BlockAccumulator::ToolUse {
					id,
					name,
					input_json,
				} => {
					let input = if input_json.is_empty() {
						Value::Object(Default::default())
					} else {
						serde_json::from_str(&input_json).map_err(|e| {
							AnthropicError::InvalidLlmResponse(format!(
								"invalid tool input JSON for '{name}': {e}"
							))
						})?
					};

					output.push(llms::Output::ToolCall {
						id,
						name,
						input,
						context: None,
					});
				}
			}
		}
		let usage = self.usage.take().ok_or_else(|| {
			AnthropicError::InvalidLlmResponse(
				"missing usage in response".into(),
			)
		})?;

		Ok(llms::Response { output, usage })
	}
}

impl LlmResponseStream for ResponseStream {
	async fn next(
		&mut self,
	) -> Option<Result<llms::LlmResponseEvent, LlmsError>> {
		if self.done {
			return None;
		}

		loop {
			let ev = match self.next_event().await {
				Some(Ok(ev)) => ev,
				Some(Err(e)) => return Some(Err(e.into())),
				None => return None,
			};

			match ev {
				Event::MessageStart { message } => {
					if let Some(usage) = message.usage {
						self.usage
							.get_or_insert_with(llms::Usage::default)
							.input_tokens = usage.input_tokens;
					}
					continue;
				}
				Event::MessageDelta { usage, .. } => {
					if let Some(usage) = usage {
						self.usage
							.get_or_insert_with(llms::Usage::default)
							.output_tokens = usage.output_tokens;
					}
					continue;
				}
				Event::ContentBlockStart { content_block, .. } => {
					let block = match content_block {
						ContentBlockStartData::Text { text } => {
							BlockAccumulator::Text { text }
						}
						ContentBlockStartData::ToolUse { id, name } => {
							BlockAccumulator::ToolUse {
								id,
								name,
								input_json: String::new(),
							}
						}
						ContentBlockStartData::Thinking { .. } => {
							BlockAccumulator::Thinking
						}
					};

					self.blocks.push(block);
					continue;
				}
				Event::ContentBlockDelta { index, delta } => {
					let acc = self.blocks.get_mut(index as usize).expect(
						"received delta for non-existent content block",
					);

					match (delta, acc) {
						(
							ContentDelta::TextDelta { text },
							BlockAccumulator::Text { text: acc },
						) => {
							acc.push_str(&text);

							return Some(Ok(
								llms::LlmResponseEvent::TextDelta {
									content: text,
								},
							));
						}
						(
							ContentDelta::InputJsonDelta { partial_json },
							BlockAccumulator::ToolUse { input_json, .. },
						) => {
							input_json.push_str(&partial_json);
							continue;
						}
						(
							ContentDelta::ThinkingDelta { .. }
							| ContentDelta::SignatureDelta { .. },
							BlockAccumulator::Thinking,
						) => continue,
						_ => unreachable!(
							"received delta of wrong type for content block"
						),
					}
				}
				Event::MessageStop => {
					self.done = true;
					let response = self
						.build_response()
						.map(llms::LlmResponseEvent::Completed)
						.map_err(Into::into);
					return Some(response);
				}
				Event::Error { error } => {
					self.done = true;
					return Some(Err(LlmsError::Response {
						status: StatusCode::OK,
						body: format!(
							"{}: {}",
							error.error_type, error.message
						),
					}));
				}
				_ => continue,
			}
		}
	}
}
