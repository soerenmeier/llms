use std::fmt;

use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tracing::trace;

use crate::{
	llms::{self, LlmProvider, LlmResponseStream, LlmsError},
	utils::{
		default_parameters,
		sse::{SseError, SseResponse},
	},
};

const ANTHROPIC_VERSION: &str = "2023-06-01";
const DEFAULT_MAX_TOKENS: u32 = 8096;

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
			stream: bool,
		}

		let api_req = ApiReq {
			model: req.model.as_str(),
			max_tokens: req.max_tokens,
			system: req.system.as_deref(),
			messages: &req.messages,
			tools: &req.tools,
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
		let model = match req.model {
			llms::Model::ClaudeOpus4_6 => AnthropicModel::Opus4_6,
			llms::Model::ClaudeSonnet4_6 => AnthropicModel::Sonnet4_6,
			llms::Model::ClaudeHaiku4_5 => AnthropicModel::Haiku4_5,
			m => unreachable!("unsupported model: {m:?}"),
		};

		let system = if req.instructions.is_empty() {
			None
		} else {
			Some(req.instructions.clone())
		};

		self.request(&Request {
			messages: req.input.iter().cloned().map(Into::into).collect(),
			model,
			system,
			tools: req.tools.iter().cloned().map(Into::into).collect(),
			max_tokens: DEFAULT_MAX_TOKENS,
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
	pub max_tokens: u32,
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
	Opus4_6,
	Sonnet4_6,
	Haiku4_5,
}

impl AnthropicModel {
	pub fn as_str(&self) -> &'static str {
		match self {
			AnthropicModel::Opus4_6 => "claude-opus-4-6",
			AnthropicModel::Sonnet4_6 => "claude-sonnet-4-6",
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
	Text { text: String },
	ToolUse { id: String, name: String },
}

#[derive(Debug, Deserialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentDelta {
	TextDelta { text: String },
	InputJsonDelta { partial_json: String },
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
}

pub struct ResponseStream {
	inner: SseResponse,
	/// Content blocks accumulated in arrival order (Anthropic always sends
	/// them sequentially, so index == position in this Vec).
	blocks: Vec<BlockAccumulator>,
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
				BlockAccumulator::ToolUse {
					id,
					name,
					input_json,
				} => {
					let input =
						serde_json::from_str(&input_json).map_err(|e| {
							AnthropicError::InvalidLlmResponse(format!(
								"invalid tool input JSON for '{name}': {e}"
							))
						})?;
					output.push(llms::Output::ToolCall {
						id,
						name,
						input,
						context: None,
					});
				}
			}
		}
		Ok(llms::Response { output })
	}
}

impl LlmResponseStream for ResponseStream {
	async fn next(&mut self) -> Option<Result<llms::ResponseEvent, LlmsError>> {
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

							return Some(Ok(llms::ResponseEvent::TextDelta {
								content: text,
							}));
						}
						(
							ContentDelta::InputJsonDelta { partial_json },
							BlockAccumulator::ToolUse { input_json, .. },
						) => {
							input_json.push_str(&partial_json);
							continue;
						}
						_ => unreachable!(
							"received delta of wrong type for content block"
						),
					}
				}
				Event::MessageStop => {
					self.done = true;
					let response = self
						.build_response()
						.map(llms::ResponseEvent::Completed)
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
