use eventsource_stream::Eventsource;
use futures::StreamExt;
use reqwest::{
	Client, StatusCode,
	header::{ACCEPT, CONTENT_TYPE, HeaderValue},
};
use serde::{Deserialize, Serialize};

use crate::utils::sse::{SseError, SseResponse};

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

		eprintln!("req {}", serde_json::to_string(&req).unwrap());

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

		let mut stream = SseResponse::new(resp);

		// while let Some(thing) = stream.next::<Event>().await {
		// 	println!("{:?}", thing);
		// }

		Ok(ResponseStream::new(stream))
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
	Custom { name: String, description: String },
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Input {
	Message(InputMessage),
	Reasoning(ReasoningItem),
	CustomToolCall(CustomToolCall),
	CustomToolCallOutput(CustomToolCallOutput),
}

impl From<OutputItem> for Input {
	fn from(item: OutputItem) -> Self {
		match item {
			OutputItem::Message(msg) => {
				Input::Message(InputMessage::Output(msg))
			}
			OutputItem::Reasoning(reasoning) => Input::Reasoning(reasoning),
			OutputItem::CustomToolCall(tool_call) => {
				Input::CustomToolCall(tool_call)
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

#[derive(Debug, thiserror::Error)]
pub enum OpenAiError {
	#[error("Response error: status {status}, body {body}")]
	ResponseError { status: StatusCode, body: String },
	#[error("Reqwest error: {0}")]
	ReqwestError(#[from] reqwest::Error),
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
pub enum OpenAiModel {
	#[serde(rename = "gpt-5")]
	Gpt5,
}

impl OpenAiModel {
	pub fn as_str(&self) -> &'static str {
		match self {
			OpenAiModel::Gpt5 => "gpt-5",
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
	#[serde(rename = "response.custom_tool_call_input.delta")]
	ResponseCustomToolCallInputDelta {
		output_index: u32,
		item_id: String,
		delta: String,
	},
	#[serde(rename = "response.custom_tool_call_input.done")]
	ResponseCustomToolCallInputDone {
		output_index: u32,
		item_id: String,
		input: String,
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
	CustomToolCall(CustomToolCall),
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
pub struct CustomToolCall {
	pub id: String,
	pub call_id: String,
	pub input: String,
	pub name: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CustomToolCallOutput {
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

pub struct ResponseStream {
	inner: SseResponse,
	pub completed_response: Option<Response>,
}

impl ResponseStream {
	fn new(inner: SseResponse) -> Self {
		Self {
			inner,
			completed_response: None,
		}
	}

	pub async fn next(&mut self) -> Option<Result<Event, SseError>> {
		let ev = match self.inner.next().await {
			Some(Ok(ev)) => ev,
			a => return a,
		};

		match &ev {
			Event::ResponseCompleted { response } => {
				self.completed_response = Some(response.clone());
			}
			_ => {}
		}

		Some(Ok(ev))
	}
}
