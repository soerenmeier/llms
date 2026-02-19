use std::mem;

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

const BASE_URL: &str =
	"https://generativelanguage.googleapis.com/v1beta/models";

pub struct Google {
	pub client: Client,
	pub api_key: String,
}

impl Google {
	pub fn new(api_key: String) -> Self {
		Self {
			client: Client::new(),
			api_key,
		}
	}

	pub async fn request(
		&self,
		req: &Request,
	) -> Result<ResponseStream, GoogleError> {
		#[derive(Debug, Serialize)]
		#[serde(rename_all = "camelCase")]
		struct ApiReq<'a> {
			contents: &'a Vec<ApiContent>,
			#[serde(skip_serializing_if = "Option::is_none")]
			system_instruction: Option<ApiSystemInstruction>,
			#[serde(skip_serializing_if = "Vec::is_empty")]
			tools: &'a Vec<ApiTool>,
		}

		#[derive(Debug, Serialize)]
		struct ApiSystemInstruction {
			parts: Vec<ApiPart>,
		}

		let system_instruction =
			req.system_instruction.as_deref().map(|text| {
				ApiSystemInstruction {
					parts: vec![ApiPart::Text {
						text: text.to_string(),
					}],
				}
			});

		let api_req = ApiReq {
			contents: &req.contents,
			system_instruction,
			tools: &req.tools,
		};

		trace!("{:?}", serde_json::to_string(&api_req));

		let url = format!(
			"{}/{}:streamGenerateContent?alt=sse",
			BASE_URL,
			req.model.as_str(),
		);

		let resp = self
			.client
			.post(&url)
			.header("x-goog-api-key", &self.api_key)
			.json(&api_req)
			.send()
			.await?;

		if !resp.status().is_success() {
			let status = resp.status();
			let body = resp.text().await?;
			return Err(GoogleError::ResponseError { status, body });
		}

		Ok(ResponseStream::new(SseResponse::new(resp)))
	}
}

impl LlmProvider for Google {
	type Stream = ResponseStream;

	async fn request(
		&self,
		req: &llms::Request,
	) -> Result<Self::Stream, LlmsError> {
		let model = match req.model {
			llms::Model::GeminiPro3 => GeminiModel::Pro3,
			llms::Model::GeminiFlash3 => GeminiModel::Flash3,
			m => unreachable!("unsupported model: {m:?}"),
		};

		let system_instruction = if req.instructions.is_empty() {
			None
		} else {
			Some(req.instructions.clone())
		};

		self.request(&Request {
			contents: req.input.iter().cloned().map(Into::into).collect(),
			model,
			system_instruction,
			tools: req.tools.iter().cloned().map(Into::into).collect(),
		})
		.await
		.map_err(Into::into)
	}
}

#[derive(Debug)]
pub struct Request {
	pub contents: Vec<ApiContent>,
	pub model: GeminiModel,
	pub system_instruction: Option<String>,
	pub tools: Vec<ApiTool>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ApiRole {
	User,
	Model,
}

impl From<llms::Role> for ApiRole {
	fn from(role: llms::Role) -> Self {
		match role {
			llms::Role::User => ApiRole::User,
			llms::Role::Assistant => ApiRole::Model,
		}
	}
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ApiContent {
	pub role: ApiRole,
	pub parts: Vec<ApiPart>,
}

impl From<llms::Input> for ApiContent {
	fn from(input: llms::Input) -> Self {
		match input {
			llms::Input::Text { role, content } => ApiContent {
				role: role.into(),
				parts: vec![ApiPart::Text { text: content }],
			},
			llms::Input::ToolCall {
				name,
				input,
				context,
				..
			} => ApiContent {
				role: ApiRole::Model,
				parts: vec![ApiPart::FunctionCall {
					function_call: ApiFunctionCall { name, args: input },
					thought_signature: context,
				}],
			},
			llms::Input::ToolCallOutput { id, output } => ApiContent {
				role: ApiRole::User,
				parts: vec![ApiPart::FunctionResponse {
					function_response: ApiFunctionResponse {
						// Gemini identifies responses by function name.
						// Output::ToolCall sets id == name, so the id here
						// is already the function name.
						name: id,
						response: serde_json::json!({ "output": output }),
					},
				}],
			},
		}
	}
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all_fields = "camelCase")]
#[serde(untagged)]
pub enum ApiPart {
	Text {
		text: String,
	},
	FunctionCall {
		function_call: ApiFunctionCall,
		/// Gemini 3 thought signature – must be echoed back exactly as received.
		/// Stored as the generic `context` field in [`llms::Input::ToolCall`].
		#[serde(skip_serializing_if = "Option::is_none")]
		thought_signature: Option<String>,
	},
	FunctionResponse {
		function_response: ApiFunctionResponse,
	},
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiFunctionCall {
	pub name: String,
	/// The arguments as a JSON object.
	pub args: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiFunctionResponse {
	pub name: String,
	/// The response payload as a JSON object.
	pub response: Value,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ApiTool {
	pub function_declarations: Vec<ApiFunctionDeclaration>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ApiFunctionDeclaration {
	pub name: String,
	pub description: String,
	/// JSON Schema object describing the function's parameters.
	/// Must be `{ "type": "object", "properties": { … } }`.
	pub parameters: Value,
}

impl From<llms::Tool> for ApiTool {
	fn from(tool: llms::Tool) -> Self {
		ApiTool {
			function_declarations: vec![ApiFunctionDeclaration {
				name: tool.name,
				description: tool.description,
				parameters: tool.parameters.unwrap_or_else(default_parameters),
			}],
		}
	}
}

#[derive(Debug, Clone, Copy)]
pub enum GeminiModel {
	Pro3,
	Flash3,
}

impl GeminiModel {
	pub fn as_str(&self) -> &'static str {
		match self {
			GeminiModel::Pro3 => "gemini-3-pro-preview",
			GeminiModel::Flash3 => "gemini-3-flash-preview",
		}
	}
}

/// A single streaming chunk from Gemini's `streamGenerateContent` endpoint.
/// Each SSE `data:` event carries a full `GenerateContentResponse` JSON object.
#[derive(Debug, Deserialize, Clone)]
pub struct StreamChunk {
	#[serde(default)]
	pub candidates: Vec<Candidate>,
	pub error: Option<ApiErrorBody>,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct Candidate {
	pub content: Option<CandidateContent>,
	pub finish_reason: Option<FinishReason>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct CandidateContent {
	pub parts: Vec<CandidatePart>,
}

/// Response parts deserialized from the model's candidate content.
///
/// Uses `#[serde(untagged)]` for the same reason as `ApiPart`: the wire
/// format discriminates by field name rather than a `"type"` tag.
/// `FunctionCall` is tried first so that an object carrying a `functionCall`
/// key is never mistakenly matched as plain text.
#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all_fields = "camelCase")]
#[serde(untagged)]
pub enum CandidatePart {
	FunctionCall {
		function_call: ApiFunctionCall,
		/// Gemini 3 thought signature. Mapped to the generic `context` field in
		/// [`llms::Output::ToolCall`] and must be round-tripped back until the
		/// corresponding tool response has been added to the history.
		thought_signature: Option<String>,
	},
	Text {
		text: String,
	},
}

#[derive(Debug, Deserialize, Clone, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum FinishReason {
	Stop,
	MaxTokens,
	Safety,
	Recitation,
	Other,
	#[serde(other)]
	Unknown,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ApiErrorBody {
	pub code: Option<u32>,
	pub message: String,
	pub status: Option<String>,
}

#[derive(Debug, thiserror::Error)]
pub enum GoogleError {
	#[error("Invalid LLM response: {0}")]
	InvalidLlmResponse(String),
	#[error("Response error: status {status}, body {body}")]
	ResponseError { status: StatusCode, body: String },
	#[error("API error {code}: {message}")]
	ApiError { code: u32, message: String },
	#[error("Reqwest error: {0}")]
	ReqwestError(#[from] reqwest::Error),
}

impl From<GoogleError> for LlmsError {
	fn from(e: GoogleError) -> Self {
		match e {
			GoogleError::InvalidLlmResponse(msg) => LlmsError::Response {
				status: StatusCode::OK,
				body: msg,
			},
			GoogleError::ResponseError { status, body } => {
				LlmsError::Response { status, body }
			}
			GoogleError::ApiError { code, message } => LlmsError::Response {
				status: StatusCode::from_u16(code as u16)
					.unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
				body: message,
			},
			GoogleError::ReqwestError(e) => LlmsError::Reqwest(e),
		}
	}
}

pub struct ResponseStream {
	inner: SseResponse,
	text_acc: String,
	tool_calls: Vec<llms::Output>,
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
			text_acc: String::new(),
			tool_calls: Vec::new(),
			done: false,
		}
	}

	async fn next_chunk(&mut self) -> Option<Result<StreamChunk, SseError>> {
		match self.inner.next().await {
			Some(Ok(chunk)) => {
				trace!("google chunk: {chunk:?}");
				Some(Ok(chunk))
			}
			other => other,
		}
	}

	fn build_response(&mut self) -> llms::Response {
		let mut output: Vec<llms::Output> =
			Vec::with_capacity(self.tool_calls.len() + 1 /* text */);

		if !self.text_acc.is_empty() {
			output.push(llms::Output::Text {
				content: mem::take(&mut self.text_acc),
			});
		}

		output.extend(self.tool_calls.drain(..));

		llms::Response { output }
	}
}

impl LlmResponseStream for ResponseStream {
	async fn next(&mut self) -> Option<Result<llms::ResponseEvent, LlmsError>> {
		if self.done {
			return None;
		}

		loop {
			let chunk = match self.next_chunk().await {
				Some(Ok(c)) => c,
				Some(Err(e)) => return Some(Err(e.into())),
				None => {
					// Stream ended without a finish_reason — treat as complete.
					self.done = true;
					return Some(Ok(llms::ResponseEvent::Completed(
						self.build_response(),
					)));
				}
			};

			if let Some(err) = chunk.error {
				self.done = true;
				return Some(Err(GoogleError::ApiError {
					code: err.code.unwrap_or(0),
					message: err.message,
				}
				.into()));
			}

			// We only ever inspect the first candidate.
			let Some(candidate) = chunk.candidates.into_iter().next() else {
				continue;
			};

			let mut text_delta = String::new();

			if let Some(content) = candidate.content {
				for part in content.parts {
					match part {
						CandidatePart::Text { text } => {
							text_delta.push_str(&text);
							self.text_acc.push_str(&text);
						}
						CandidatePart::FunctionCall {
							function_call,
							thought_signature,
						} => {
							self.tool_calls.push(llms::Output::ToolCall {
								// Gemini has no separate opaque call id.
								// We use the function name for both fields so
								// that ToolCallOutput can round-trip via id.
								id: function_call.name.clone(),
								name: function_call.name,
								input: function_call.args,
								context: thought_signature,
							});
						}
					}
				}
			}

			// On the final chunk, emit Completed (which carries all
			// accumulated text + tool calls) regardless of whether this chunk
			// also contained a text delta — the delta is already in text_acc.
			if candidate.finish_reason.is_some() {
				self.done = true;
				return Some(Ok(llms::ResponseEvent::Completed(
					self.build_response(),
				)));
			}

			if !text_delta.is_empty() {
				return Some(Ok(llms::ResponseEvent::TextDelta {
					content: text_delta,
				}));
			}
		}
	}
}
