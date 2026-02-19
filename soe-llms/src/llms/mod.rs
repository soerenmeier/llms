pub mod error;

pub use error::LlmsError;

use crate::openai;

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
		input: String,
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
	Developer,
	User,
	Assistant,
}

#[derive(Debug, Clone, Copy)]
pub enum Model {
	Gpt5,
	Gpt5Mini,
	Gpt5Nano,
	Gpt5_2,
}

#[derive(Debug, Clone)]
pub struct Tool {
	pub name: String,
	pub description: String,
}

pub struct LlmsConfig {
	pub openai_api_key: Option<String>,
}

struct LlmProviders {
	open_ai: Option<openai::OpenAi>,
}

pub struct Llms {
	inner: LlmProviders,
}

impl Llms {
	pub fn new(config: LlmsConfig) -> Self {
		Self {
			inner: LlmProviders {
				open_ai: config.openai_api_key.map(openai::OpenAi::new),
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
	// add usage?
}

#[derive(Debug)]
pub enum Output {
	Text {
		content: String,
	},
	ToolCall {
		id: String,
		name: String,
		input: String,
	},
}

#[derive(Debug)]
pub struct ResponseStream {
	inner: RespStreamInner,
}

#[derive(Debug)]
enum RespStreamInner {
	OpenAi(openai::ResponseStream),
}

impl ResponseStream {
	pub async fn next(&mut self) -> Option<Result<ResponseEvent, LlmsError>> {
		use RespStreamInner::*;

		match &mut self.inner {
			OpenAi(stream) => LlmResponseStream::next(stream).await,
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
