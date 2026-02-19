use reqwest::{
	Client, StatusCode,
	header::{ACCEPT, HeaderValue},
};
use serde::{Deserialize, Serialize};
use tracing::{trace, warn};

use crate::{
	llms::{self, LlmProvider, LlmResponseStream, LlmsError},
	utils::{
		default_parameters,
		sse::{SseError, SseResponse},
	},
};

pub struct OpenAi {
	pub client: Client,
	pub api_key: String,
}

impl OpenAi {
	pub fn new(api_key: String) -> Self {
		Self {
			client: Client::new(),
			api_key,
		}
	}

	pub async fn request(
		&self,
		req: &Request,
	) -> Result<ResponseStream, OpenAiError> {
		#[derive(Debug, Serialize)]
		struct Req<'a> {
			input: &'a Vec<Input>,
			instructions: &'a String,
			model: OpenAiModel,
			prompt_cache_key: &'a String,
			safety_identifier: &'a String,
			tools: &'a Vec<Tool>,
			stream: bool,
		}

		let req = Req {
			input: &req.input,
			instructions: &req.instructions,
			model: req.model,
			prompt_cache_key: &req.prompt_cache_key,
			safety_identifier: &req.safety_identifier,
			tools: &req.tools,
			stream: true,
		};

		trace!("{:?}", req);

		let resp = self
			.client
			.post("https://api.openai.com/v1/responses")
			.bearer_auth(&self.api_key)
			.header(ACCEPT, HeaderValue::from_static("text/event-stream"))
			.json(&req)
			.send()
			.await?;

		if !resp.status().is_success() {
			let status = resp.status();
			let body = resp.text().await?;

			return Err(OpenAiError::ResponseError { status, body });
		}

		Ok(ResponseStream::new(SseResponse::new(resp)))
	}
}

impl LlmProvider for OpenAi {
	type Stream = ResponseStream;

	async fn request(
		&self,
		req: &llms::Request,
	) -> Result<Self::Stream, LlmsError> {
		let model = match &req.model {
			llms::Model::Gpt5 => OpenAiModel::Gpt5,
			llms::Model::Gpt5Mini => OpenAiModel::Gpt5Mini,
			llms::Model::Gpt5Nano => OpenAiModel::Gpt5Nano,
			llms::Model::Gpt5_2 => OpenAiModel::Gpt5_2,
			m => unreachable!("unsupported model: {m:?}"),
		};

		self.request(&Request {
			input: req.input.iter().cloned().map(Into::into).collect(),
			instructions: req.instructions.clone(),
			model,
			prompt_cache_key: req.user_id.clone(),
			safety_identifier: req.user_id.clone(),
			tools: req.tools.iter().cloned().map(Into::into).collect(),
		})
		.await
		.map_err(Into::into)
	}
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Request {
	pub input: Vec<Input>,
	pub instructions: String,
	pub model: OpenAiModel,
	pub prompt_cache_key: String,
	pub safety_identifier: String,
	pub tools: Vec<Tool>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Tool {
	Function {
		name: String,
		#[serde(skip_serializing_if = "Option::is_none")]
		description: Option<String>,
		/// standard JSON Schema object
		/// (`{ "type": "object", "properties": { … }, "required": […] }`).
		parameters: serde_json::Value,
		strict: bool,
	},
}

impl From<llms::Tool> for Tool {
	fn from(tool: llms::Tool) -> Self {
		Tool::Function {
			name: tool.name,
			description: Some(tool.description).filter(|d| !d.is_empty()),
			parameters: tool.parameters.unwrap_or_else(default_parameters),
			strict: false,
		}
	}
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Input {
	Message(InputMessage),
	Reasoning(ReasoningItem),
	FunctionCall(FunctionCall),
	FunctionCallOutput(FunctionCallOutput),
}

impl From<OutputItem> for Input {
	fn from(item: OutputItem) -> Self {
		match item {
			OutputItem::Message(msg) => {
				Input::Message(InputMessage::Output(msg))
			}
			OutputItem::Reasoning(reasoning) => Input::Reasoning(reasoning),
			OutputItem::FunctionCall(tool_call) => {
				Input::FunctionCall(tool_call)
			}
		}
	}
}

impl From<llms::Input> for Input {
	fn from(input: llms::Input) -> Self {
		match input {
			llms::Input::Text { role, content } => {
				Input::Message(InputMessage::Input {
					role: role.into(),
					content,
				})
			}
			llms::Input::ToolCall { id, name, input } => {
				Input::FunctionCall(FunctionCall {
					id: None,
					call_id: id,
					name,
					arguments: input.to_string(),
					status: None,
				})
			}
			llms::Input::ToolCallOutput { id, output } => {
				Input::FunctionCallOutput(FunctionCallOutput {
					id: None,
					call_id: id,
					output,
				})
			}
		}
	}
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(untagged)]
pub enum InputMessage {
	// first because of serde untagged priority
	Output(OutputMessage),
	Input { role: Role, content: String },
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
#[serde(rename_all = "lowercase")]
pub enum Role {
	Developer,
	User,
	Assistant,
}

impl From<llms::Role> for Role {
	fn from(role: llms::Role) -> Self {
		match role {
			llms::Role::User => Role::User,
			llms::Role::Assistant => Role::Assistant,
		}
	}
}

#[derive(Debug, thiserror::Error)]
pub enum OpenAiError {
	#[error("Invalid LLM response: {0}")]
	InvalidLlmResponse(String),
	#[error("Response error: status {status}, body {body}")]
	ResponseError { status: StatusCode, body: String },
	#[error("Reqwest error: {0}")]
	ReqwestError(#[from] reqwest::Error),
}

impl From<OpenAiError> for LlmsError {
	fn from(e: OpenAiError) -> Self {
		match e {
			OpenAiError::InvalidLlmResponse(msg) => LlmsError::Response {
				status: StatusCode::OK,
				body: msg,
			},
			OpenAiError::ResponseError { status, body } => {
				LlmsError::Response { status, body }
			}
			OpenAiError::ReqwestError(e) => LlmsError::Reqwest(e),
		}
	}
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
pub enum OpenAiModel {
	#[serde(rename = "gpt-5")]
	Gpt5,
	#[serde(rename = "gpt-5-mini")]
	Gpt5Mini,
	#[serde(rename = "gpt-5-nano")]
	Gpt5Nano,
	#[serde(rename = "gpt-5.2")]
	Gpt5_2,
}

impl OpenAiModel {
	pub fn as_str(&self) -> &'static str {
		match self {
			OpenAiModel::Gpt5 => "gpt-5",
			OpenAiModel::Gpt5Mini => "gpt-5-mini",
			OpenAiModel::Gpt5Nano => "gpt-5-nano",
			OpenAiModel::Gpt5_2 => "gpt-5.2",
		}
	}
}

impl AsRef<str> for OpenAiModel {
	fn as_ref(&self) -> &str {
		self.as_str()
	}
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type")]
pub enum Event {
	#[serde(rename = "response.created")]
	ResponseCreated { response: Response },
	#[serde(rename = "response.in_progress")]
	ResponseInProgress { response: Response },
	#[serde(rename = "response.completed")]
	ResponseCompleted { response: Response },
	#[serde(rename = "response.output_item.added")]
	ResponseOutputItemAdded { output_index: u32, item: OutputItem },
	#[serde(rename = "response.output_item.done")]
	ResponseOutputItemDone { output_index: u32, item: OutputItem },
	#[serde(rename = "response.content_part.added")]
	ResponseContentPartAdded {
		output_index: u32,
		item_id: String,
		content_index: u32,
		part: OutputMessageContent,
	},
	#[serde(rename = "response.content_part.done")]
	ResponseContentPartDone {
		output_index: u32,
		item_id: String,
		content_index: u32,
		part: OutputMessageContent,
	},
	#[serde(rename = "response.output_text.delta")]
	ResponseOutputTextDelta {
		output_index: u32,
		item_id: String,
		content_index: u32,
		delta: String,
	},
	#[serde(rename = "response.output_text.done")]
	ResponseOutputTextDone {
		output_index: u32,
		item_id: String,
		content_index: u32,
		text: String,
	},
	#[serde(rename = "response.function_call_arguments.delta")]
	ResponseFunctionCallArgumentsDelta {
		output_index: u32,
		item_id: String,
		delta: String,
	},
	#[serde(rename = "response.function_call_arguments.done")]
	ResponseFunctionCallArgumentsDone {
		output_index: u32,
		item_id: String,
		arguments: String,
	},
	#[serde(rename = "error")]
	ResponseError {
		code: Option<String>,
		message: String,
	},
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Response {
	pub output: Vec<OutputItem>,
	pub status: ResponseStatus,
	pub usage: Option<ResponseUsage>,
}

impl TryFrom<Response> for llms::Response {
	type Error = OpenAiError;

	fn try_from(resp: Response) -> Result<Self, Self::Error> {
		if !matches!(resp.status, ResponseStatus::Completed) {
			return Err(OpenAiError::InvalidLlmResponse(format!(
				"response status is not completed: {:?}",
				resp.status
			)));
		}

		Ok(llms::Response {
			output: resp
				.output
				.into_iter()
				.filter_map(|o| Option::<llms::Output>::try_from(o).transpose())
				.collect::<Result<_, Self::Error>>()?,
		})
	}
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ResponseStatus {
	Completed,
	Failed,
	InProgress,
	Cancelled,
	Queued,
	Incomplete,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum OutputItem {
	Message(OutputMessage),
	Reasoning(ReasoningItem),
	FunctionCall(FunctionCall),
}

impl TryFrom<OutputItem> for Option<llms::Output> {
	type Error = OpenAiError;

	fn try_from(item: OutputItem) -> Result<Self, OpenAiError> {
		match item {
			OutputItem::Message(msg) => {
				assert!(matches!(msg.status, OutputStatus::Completed));

				if msg.content.len() > 1 {
					warn!("output message has multiple items");
				}

				Ok(Some(llms::Output::Text {
					content: msg
						.content
						.into_iter()
						.filter_map(|c| match c {
							OutputMessageContent::OutputText { text } => {
								Some(text)
							}
							OutputMessageContent::Refusal { refusal } => {
								Some(refusal)
							}
							_ => None,
						})
						.collect(),
				}))
			}
			OutputItem::Reasoning(_) => Ok(None),
			OutputItem::FunctionCall(fc) => {
				assert!(matches!(fc.status, Some(OutputStatus::Completed)));

				let input =
					serde_json::from_str(&fc.arguments).map_err(|e| {
						OpenAiError::InvalidLlmResponse(format!(
							"failed to parse function call arguments: {e}"
						))
					})?;

				Ok(Some(llms::Output::ToolCall {
					id: fc.call_id,
					name: fc.name,
					input,
				}))
			}
		}
	}
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct OutputMessage {
	pub id: String,
	pub status: OutputStatus,
	pub role: Role,
	pub content: Vec<OutputMessageContent>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OutputMessageContent {
	OutputText { text: String },
	Refusal { refusal: String },
	ReasoningText { text: String },
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
#[serde(rename_all = "snake_case")]
pub enum OutputStatus {
	InProgress,
	Completed,
	Incomplete,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ReasoningItem {
	pub id: String,
	pub summary: Vec<ReasoningSummary>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub status: Option<OutputStatus>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ReasoningSummary {
	SummaryText { text: String },
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FunctionCall {
	pub id: Option<String>,
	pub call_id: String,
	pub name: String,
	/// JSON string of the arguments chosen by the model.
	pub arguments: String,
	pub status: Option<OutputStatus>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FunctionCallOutput {
	#[serde(skip_serializing_if = "Option::is_none")]
	pub id: Option<String>,
	pub call_id: String,
	pub output: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ResponseUsage {
	pub input_tokens: u32,
	pub output_tokens: u32,
	pub total_tokens: u32,
}

#[derive(Debug)]
pub struct ResponseStream {
	inner: SseResponse,
}

impl ResponseStream {
	fn new(inner: SseResponse) -> Self {
		Self { inner }
	}

	pub async fn next(&mut self) -> Option<Result<Event, SseError>> {
		match self.inner.next().await {
			Some(Ok(ev)) => {
				trace!("new event: {ev:?}");
				Some(Ok(ev))
			}
			a => return a,
		}
	}
}

impl LlmResponseStream for ResponseStream {
	async fn next(&mut self) -> Option<Result<llms::ResponseEvent, LlmsError>> {
		loop {
			let ev = match self.next().await {
				Some(Ok(ev)) => ev,
				Some(Err(e)) => return Some(Err(e.into())),
				None => return None,
			};

			break Some(match ev {
				Event::ResponseOutputTextDelta { delta, .. } => {
					Ok(llms::ResponseEvent::TextDelta { content: delta })
				}
				Event::ResponseCompleted { response } => response
					.try_into()
					.map(llms::ResponseEvent::Completed)
					.map_err(Into::into),
				Event::ResponseError { code, message } => {
					return Some(Err(LlmsError::Response {
						status: StatusCode::OK,
						body: format!("{code:?}: {message}",),
					}));
				}
				_ => continue,
			});
		}
	}
}
