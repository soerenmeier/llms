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

pub struct Mistral {
	pub client: Client,
	pub api_key: String,
}

impl Mistral {
	pub fn new(api_key: String) -> Self {
		Self {
			client: Client::new(),
			api_key,
		}
	}

	pub async fn request(
		&self,
		req: &Request,
	) -> Result<ResponseStream, MistralError> {
		#[derive(Debug, Serialize)]
		struct ApiReq<'a> {
			model: &'a str,
			messages: &'a Vec<ApiMessage>,
			#[serde(skip_serializing_if = "Vec::is_empty")]
			tools: &'a Vec<ApiTool>,
			stream: bool,
		}

		let api_req = ApiReq {
			model: req.model.as_str(),
			messages: &req.messages,
			tools: &req.tools,
			stream: true,
		};

		trace!("{:?}", serde_json::to_string(&api_req));

		let resp = self
			.client
			.post("https://api.mistral.ai/v1/chat/completions")
			.bearer_auth(&self.api_key)
			.json(&api_req)
			.send()
			.await?;

		if !resp.status().is_success() {
			let status = resp.status();
			let body = resp.text().await?;
			return Err(MistralError::ResponseError { status, body });
		}

		Ok(ResponseStream::new(SseResponse::new(resp)))
	}
}

impl LlmProvider for Mistral {
	type Stream = ResponseStream;

	async fn request(
		&self,
		req: &llms::Request,
	) -> Result<Self::Stream, LlmsError> {
		let model = match req.model {
			llms::Model::MistralLarge3 => MistralModel::Large3,
			llms::Model::MistralMedium3_1 => MistralModel::Medium3_1,
			llms::Model::MistralSmall3_2 => MistralModel::Small3_2,
			llms::Model::Devstral2 => MistralModel::Devstral2,
			llms::Model::MagistralMedium1_2 => MistralModel::MagistralMedium1_2,
			m => unreachable!("unsupported model: {m:?}"),
		};

		let mut messages: Vec<ApiMessage> = Vec::new();

		if !req.instructions.is_empty() {
			messages.push(ApiMessage::System {
				content: req.instructions.clone(),
			});
		}

		messages.extend(req.input.iter().cloned().map(ApiMessage::from));

		self.request(&Request {
			messages,
			model,
			tools: req.tools.iter().cloned().map(Into::into).collect(),
		})
		.await
		.map_err(Into::into)
	}
}

pub struct Request {
	pub messages: Vec<ApiMessage>,
	pub model: MistralModel,
	pub tools: Vec<ApiTool>,
}

#[derive(Debug, Clone, Copy)]
pub enum MistralModel {
	Large3,
	Medium3_1,
	Small3_2,
	Devstral2,
	MagistralMedium1_2,
}

impl MistralModel {
	pub fn as_str(&self) -> &'static str {
		match self {
			MistralModel::Large3 => "mistral-large-2512",
			MistralModel::Medium3_1 => "mistral-medium-2508",
			MistralModel::Small3_2 => "mistral-small-2506",
			MistralModel::Devstral2 => "devstral-2512",
			MistralModel::MagistralMedium1_2 => "magistral-medium-2509",
		}
	}
}

#[derive(Debug, Serialize)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum ApiMessage {
	System {
		content: String,
	},
	User {
		content: String,
	},
	Assistant {
		#[serde(skip_serializing_if = "Option::is_none")]
		content: Option<String>,
		#[serde(skip_serializing_if = "Option::is_none")]
		tool_calls: Option<Vec<ApiToolCall>>,
	},
	Tool {
		tool_call_id: String,
		content: String,
	},
}

impl From<llms::Input> for ApiMessage {
	fn from(input: llms::Input) -> Self {
		match input {
			llms::Input::Text { role, content } => match role {
				llms::Role::User => ApiMessage::User { content },
				llms::Role::Assistant => ApiMessage::Assistant {
					content: Some(content),
					tool_calls: None,
				},
			},
			llms::Input::ToolCall {
				id, name, input, ..
			} => ApiMessage::Assistant {
				content: None,
				tool_calls: Some(vec![ApiToolCall {
					id,
					kind: "function".into(),
					function: ApiToolCallFunction {
						name,
						arguments: input.to_string(),
					},
				}]),
			},
			llms::Input::ToolCallOutput { id, output } => ApiMessage::Tool {
				tool_call_id: id,
				content: output,
			},
		}
	}
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ApiToolCall {
	pub id: String,
	#[serde(rename = "type")]
	pub kind: String,
	pub function: ApiToolCallFunction,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ApiToolCallFunction {
	pub name: String,
	/// JSON-encoded arguments string.
	pub arguments: String,
}

#[derive(Debug, Serialize)]
pub struct ApiTool {
	#[serde(rename = "type")]
	pub kind: String,
	pub function: ApiToolFunction,
}

#[derive(Debug, Serialize)]
pub struct ApiToolFunction {
	pub name: String,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub description: Option<String>,
	/// Full JSON Schema object (`{ "type": "object", "properties": { … } }`).
	pub parameters: Value,
}

impl From<llms::Tool> for ApiTool {
	fn from(tool: llms::Tool) -> Self {
		ApiTool {
			kind: "function".into(),
			function: ApiToolFunction {
				name: tool.name,
				description: Some(tool.description).filter(|d| !d.is_empty()),
				parameters: tool.parameters.unwrap_or_else(default_parameters),
			},
		}
	}
}

#[derive(Debug, Deserialize)]
pub struct Chunk {
	pub choices: Vec<ChunkChoice>,
}

#[derive(Debug, Deserialize)]
pub struct ChunkChoice {
	pub delta: Delta,
}

#[derive(Debug, Deserialize)]
pub struct Delta {
	pub content: Option<DeltaContent>,
	pub tool_calls: Option<Vec<ToolCallDelta>>,
}

/// `content` is a plain string on standard models, or an array of typed blocks
/// on Magistral reasoning models (which interleave `"thinking"` and `"text"`).
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum DeltaContent {
	Text(String),
	Blocks(Vec<ContentBlock>),
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
	Text { text: String },
	Thinking { thinking: Vec<ThinkingBlock> },
}

#[derive(Debug, Deserialize)]
pub struct ThinkingBlock {
	pub text: String,
}

impl DeltaContent {
	/// Extracts the text, ignoring `"thinking"` and any other non-text blocks.
	/// Returns `None` if the result is empty.
	pub fn into_text(self) -> Option<String> {
		let text = match self {
			DeltaContent::Text(s) => s,
			DeltaContent::Blocks(blocks) => blocks
				.into_iter()
				.filter_map(|b| match b {
					ContentBlock::Text { text } => Some(text),
					ContentBlock::Thinking { .. } => None,
				})
				.collect(),
		};

		Some(text).filter(|t| !t.is_empty())
	}
}

#[derive(Debug, Deserialize)]
pub struct ToolCallDelta {
	pub index: usize,
	/// Only present on the first delta for a given slot.
	pub id: Option<String>,
	pub function: Option<ToolCallFunctionDelta>,
}

#[derive(Debug, Deserialize)]
pub struct ToolCallFunctionDelta {
	/// Only present on the first delta for a given slot.
	pub name: Option<String>,
	pub arguments: Option<String>,
}

#[derive(Debug, thiserror::Error)]
pub enum MistralError {
	#[error("Invalid LLM response: {0}")]
	InvalidLlmResponse(String),
	#[error("No output in response")]
	NoOutput,
	#[error("Response error: status {status}, body {body}")]
	ResponseError { status: StatusCode, body: String },
	#[error("Reqwest error: {0}")]
	ReqwestError(#[from] reqwest::Error),
}

impl From<MistralError> for LlmsError {
	fn from(e: MistralError) -> Self {
		match e {
			MistralError::InvalidLlmResponse(msg) => LlmsError::Response {
				status: StatusCode::OK,
				body: msg,
			},
			MistralError::NoOutput => LlmsError::Response {
				status: StatusCode::OK,
				body: "no output in response".into(),
			},
			MistralError::ResponseError { status, body } => {
				LlmsError::Response { status, body }
			}
			MistralError::ReqwestError(e) => LlmsError::Reqwest(e),
		}
	}
}

impl From<SseError> for MistralError {
	fn from(e: SseError) -> Self {
		match e {
			SseError::Reqwest(e) => MistralError::ReqwestError(e),
			other => MistralError::InvalidLlmResponse(other.to_string()),
		}
	}
}

#[derive(Default)]
struct ToolCallAccumulator {
	id: String,
	name: String,
	arguments: String,
}

pub struct ResponseStream {
	inner: SseResponse,
	/// Accumulated text across all content deltas. `None` until the first
	/// non-empty content delta arrives.
	text: Option<String>,
	/// Per-index tool call state. The index matches the `index` field in the
	/// streaming delta and grows on demand.
	tool_calls: Vec<ToolCallAccumulator>,
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
			text: None,
			tool_calls: Vec::new(),
			done: false,
		}
	}

	fn build_response(&mut self) -> Result<llms::Response, MistralError> {
		let mut output =
			Vec::with_capacity(self.tool_calls.len() + 1 /* text */);

		if let Some(text) = self.text.take() {
			output.push(llms::Output::Text { content: text });
		}

		for tc in self.tool_calls.drain(..) {
			let input = serde_json::from_str(&tc.arguments).map_err(|e| {
				MistralError::InvalidLlmResponse(format!(
					"invalid tool call arguments JSON for '{}': {e}",
					tc.name
				))
			})?;

			output.push(llms::Output::ToolCall {
				id: tc.id,
				name: tc.name,
				input,
				context: None,
			});
		}

		if output.is_empty() {
			return Err(MistralError::NoOutput);
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
			let chunk: Chunk = match self.inner.next().await {
				Some(Ok(c)) => c,
				Some(Err(e)) => return Some(Err(e.into())),
				None => {
					self.done = true;
					let response = self
						.build_response()
						.map(llms::ResponseEvent::Completed)
						.map_err(Into::into);
					return Some(response);
				}
			};

			trace!("mistral chunk: {chunk:?}");

			let choice = match chunk.choices.into_iter().next() {
				Some(c) => c,
				None => continue,
			};

			if let Some(tc_deltas) = choice.delta.tool_calls {
				for delta in tc_deltas {
					// Grow the accumulator vec on demand (indices are always
					// contiguous and arrive in order per the spec).
					self.tool_calls
						.resize_with(delta.index + 1, Default::default);

					let acc = &mut self.tool_calls[delta.index];

					if let Some(id) = delta.id {
						acc.id = id;
					}

					if let Some(func) = delta.function {
						if let Some(name) = func.name {
							acc.name = name;
						}
						if let Some(args) = func.arguments {
							acc.arguments.push_str(&args);
						}
					}
				}
			}

			if let Some(text) =
				choice.delta.content.and_then(DeltaContent::into_text)
			{
				self.text.get_or_insert_with(String::new).push_str(&text);
				return Some(Ok(llms::ResponseEvent::TextDelta {
					content: text,
				}));
			}
		}
	}
}
