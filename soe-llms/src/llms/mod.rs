pub mod error;

pub use error::LlmsError;

use crate::openai;

#[derive(Debug)]
pub struct Request {
	pub input: Vec<Input>,
	pub instructions: String,
	pub model: Model,
	pub user_id: String,
	pub tools: Vec<Tool>,
}

#[derive(Debug)]
pub struct Input {
	pub id: Option<String>,
	pub role: Role,
	pub content: String,
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
}

#[derive(Debug)]
pub struct Tool {
	name: String,
	description: String,
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
		todo!()
	}
}

pub(crate) trait LlmProvider {
	type Stream: LlmResponseStream;

	async fn request(&self, req: &Request) -> Result<Self::Stream, LlmsError>;
}

pub(crate) trait LlmResponseStream {
	async fn next(&mut self) -> Option<Result<LlmResponseEvent, LlmsError>>;
}

pub(crate) enum LlmResponseEvent {
	ResponseCompleted(Response),
}

#[derive(Debug)]
pub struct Output {}

#[derive(Debug)]
#[non_exhaustive]
pub struct Response {
	pub output: Vec<Output>,
	// add usage?
}

pub struct ResponseStream {
	inner: RespStreamInner,
}

enum RespStreamInner {
	OpenAi(openai::ResponseStream),
}

impl ResponseStream {}
