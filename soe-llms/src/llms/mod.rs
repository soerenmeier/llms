pub mod error;

pub use error::LlmsError;

use serde_json::Value;

use crate::{anthropic, openai};

#[derive(Debug, Clone)]
pub struct Request {
	pub input: Vec<Input>,
	pub instructions: String,
	pub model: Model,
	pub user_id: String,
	pub tools: Vec<Tool>,
}

#[derive(Debug, Clone)]
pub enum Input {
	Text {
		role: Role,
		content: String,
	},
	ToolCall {
		id: String,
		name: String,
		input: Value,
	},
	ToolCallOutput {
		id: String,
		output: String,
	},
}

impl From<Output> for Input {
	fn from(output: Output) -> Self {
		match output {
			Output::Text { content } => Input::Text {
				role: Role::Assistant,
				content,
			},
			Output::ToolCall { id, name, input } => {
				Input::ToolCall { id, name, input }
			}
		}
	}
}

#[derive(Debug, Clone, Copy)]
pub enum Role {
	User,
	Assistant,
}

#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum Model {
	Gpt5,
	Gpt5Mini,
	Gpt5Nano,
	Gpt5_2,
	ClaudeOpus4_6,
	ClaudeSonnet4_6,
	ClaudeHaiku4_5,
}

#[derive(Debug, Clone)]
pub struct Tool {
	pub name: String,
	pub description: String,
	/// JSON Schema object for the tool's input parameters, e.g.:
	/// ```json
	/// {
	///   "type": "object",
	///   "properties": {
	///     "location": { "type": "string", "description": "City name" }
	///   },
	///   "required": ["location"]
	/// }
	/// ```
	// None = { "type": "object", "properties": {} }
	pub parameters: Option<Value>,
}

#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct LlmsConfig {
	pub openai_api_key: Option<String>,
	pub anthropic_api_key: Option<String>,
}

impl LlmsConfig {
	pub fn new() -> Self {
		Self::default()
	}

	pub fn openai(mut self, api_key: String) -> Self {
		self.openai_api_key = Some(api_key);
		self
	}

	pub fn anthropic(mut self, api_key: String) -> Self {
		self.anthropic_api_key = Some(api_key);
		self
	}
}

struct LlmProviders {
	open_ai: Option<openai::OpenAi>,
	anthropic: Option<anthropic::Anthropic>,
}

pub struct Llms {
	inner: LlmProviders,
}

impl Llms {
	pub fn new(config: LlmsConfig) -> Self {
		Self {
			inner: LlmProviders {
				open_ai: config.openai_api_key.map(openai::OpenAi::new),
				anthropic: config
					.anthropic_api_key
					.map(anthropic::Anthropic::new),
			},
		}
	}

	pub async fn request(
		&self,
		req: &Request,
	) -> Result<ResponseStream, LlmsError> {
		match req.model {
			Model::Gpt5 | Model::Gpt5Mini | Model::Gpt5Nano | Model::Gpt5_2 => {
				let llm = self.inner.open_ai.as_ref().ok_or_else(|| {
					LlmsError::LlmNotConfigured("OpenAI".into())
				})?;
				LlmProvider::request(llm, req).await.map(Into::into)
			}
			Model::ClaudeOpus4_6
			| Model::ClaudeSonnet4_6
			| Model::ClaudeHaiku4_5 => {
				let llm = self.inner.anthropic.as_ref().ok_or_else(|| {
					LlmsError::LlmNotConfigured("Anthropic".into())
				})?;
				LlmProvider::request(llm, req).await.map(Into::into)
			}
		}
	}
}

pub(crate) trait LlmProvider {
	type Stream: LlmResponseStream;

	async fn request(&self, req: &Request) -> Result<Self::Stream, LlmsError>;
}

pub(crate) trait LlmResponseStream {
	async fn next(&mut self) -> Option<Result<ResponseEvent, LlmsError>>;
}

#[derive(Debug)]
pub enum ResponseEvent {
	TextDelta { content: String },
	Completed(Response),
}

#[derive(Debug)]
#[non_exhaustive]
pub struct Response {
	pub output: Vec<Output>,
}

#[derive(Debug)]
pub enum Output {
	Text {
		content: String,
	},
	ToolCall {
		id: String,
		name: String,
		input: Value,
	},
}

#[derive(Debug)]
pub struct ResponseStream {
	inner: RespStreamInner,
}

#[derive(Debug)]
enum RespStreamInner {
	OpenAi(openai::ResponseStream),
	Anthropic(anthropic::ResponseStream),
}

impl ResponseStream {
	pub async fn next(&mut self) -> Option<Result<ResponseEvent, LlmsError>> {
		use RespStreamInner::*;

		match &mut self.inner {
			OpenAi(stream) => LlmResponseStream::next(stream).await,
			Anthropic(stream) => LlmResponseStream::next(stream).await,
		}
	}
}

impl From<openai::ResponseStream> for ResponseStream {
	fn from(stream: openai::ResponseStream) -> Self {
		Self {
			inner: RespStreamInner::OpenAi(stream),
		}
	}
}

impl From<anthropic::ResponseStream> for ResponseStream {
	fn from(stream: anthropic::ResponseStream) -> Self {
		Self {
			inner: RespStreamInner::Anthropic(stream),
		}
	}
}
