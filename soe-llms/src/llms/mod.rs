pub mod error;

pub use error::LlmsError;

use serde_json::Value;

use crate::{anthropic, google, mistral, openai, publicai, xai};

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
		/// Opaque, provider-specific context that must be passed back with this
		/// tool call until the corresponding [`Input::ToolCallOutput`] has been
		/// added to the conversation history. After that point it can safely be
		/// discarded — the provider only validates it within the current turn.
		context: Option<String>,
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
			Output::ToolCall {
				id,
				name,
				input,
				context,
			} => Input::ToolCall {
				id,
				name,
				input,
				context,
			},
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
	GeminiPro3,
	GeminiFlash3,
	Grok4_1Fast,
	Grok4_1FastNonReasoning,
	GrokCodeFast1,
	MistralLarge3,
	MistralMedium3_1,
	MistralSmall3_2,
	Devstral2,
	MagistralMedium1_2,
	// At the moment tool calls are not supported
	Apertus8bInstruct,
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
	pub google_api_key: Option<String>,
	pub xai_api_key: Option<String>,
	pub mistral_api_key: Option<String>,
	pub publicai_api_key: Option<String>,
}

impl LlmsConfig {
	pub fn new() -> Self {
		Self::default()
	}

	pub fn openai(mut self, api_key: impl Into<Option<String>>) -> Self {
		self.openai_api_key = api_key.into();
		self
	}

	pub fn anthropic(mut self, api_key: impl Into<Option<String>>) -> Self {
		self.anthropic_api_key = api_key.into();
		self
	}

	pub fn google(mut self, api_key: impl Into<Option<String>>) -> Self {
		self.google_api_key = api_key.into();
		self
	}

	pub fn xai(mut self, api_key: impl Into<Option<String>>) -> Self {
		self.xai_api_key = api_key.into();
		self
	}

	pub fn mistral(mut self, api_key: impl Into<Option<String>>) -> Self {
		self.mistral_api_key = api_key.into();
		self
	}

	pub fn publicai(mut self, api_key: impl Into<Option<String>>) -> Self {
		self.publicai_api_key = api_key.into();
		self
	}
}

#[derive(Debug, Clone)]
struct LlmProviders {
	open_ai: Option<openai::OpenAi>,
	anthropic: Option<anthropic::Anthropic>,
	google: Option<google::Google>,
	xai: Option<xai::XAi>,
	mistral: Option<mistral::Mistral>,
	publicai: Option<publicai::PublicAi>,
}

#[derive(Debug, Clone)]
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
				google: config.google_api_key.map(google::Google::new),
				xai: config.xai_api_key.map(xai::XAi::new),
				mistral: config.mistral_api_key.map(mistral::Mistral::new),
				publicai: config.publicai_api_key.map(publicai::PublicAi::new),
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
			Model::GeminiPro3 | Model::GeminiFlash3 => {
				let llm = self.inner.google.as_ref().ok_or_else(|| {
					LlmsError::LlmNotConfigured("Google".into())
				})?;
				LlmProvider::request(llm, req).await.map(Into::into)
			}
			Model::Grok4_1Fast
			| Model::Grok4_1FastNonReasoning
			| Model::GrokCodeFast1 => {
				let llm =
					self.inner.xai.as_ref().ok_or_else(|| {
						LlmsError::LlmNotConfigured("xAI".into())
					})?;
				LlmProvider::request(llm, req).await.map(Into::into)
			}
			Model::MistralLarge3
			| Model::MistralMedium3_1
			| Model::MistralSmall3_2
			| Model::Devstral2
			| Model::MagistralMedium1_2 => {
				let llm = self.inner.mistral.as_ref().ok_or_else(|| {
					LlmsError::LlmNotConfigured("Mistral".into())
				})?;
				LlmProvider::request(llm, req).await.map(Into::into)
			}
			Model::Apertus8bInstruct => {
				let llm = self.inner.publicai.as_ref().ok_or_else(|| {
					LlmsError::LlmNotConfigured("PublicAI".into())
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
	/// Note, on some providers the last TextDelta may not be emitted
	/// but returned as part of the final Completed event instead.
	TextDelta {
		content: String,
	},
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
		/// Opaque, provider-specific context to be round-tripped into
		/// [`Input::ToolCall::context`] until the corresponding
		/// [`Input::ToolCallOutput`] is added. After that it can be discarded.
		///
		/// Currently populated by Gemini 3 thinking models (`thoughtSignature`)
		/// to preserve reasoning state across multi-step tool use within a
		/// single turn. All other providers set this to `None`.
		context: Option<String>,
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
	Google(google::ResponseStream),
	XAi(xai::ResponseStream),
	Mistral(mistral::ResponseStream),
	PublicAi(publicai::ResponseStream),
}

impl ResponseStream {
	pub async fn next(&mut self) -> Option<Result<ResponseEvent, LlmsError>> {
		use RespStreamInner::*;

		match &mut self.inner {
			OpenAi(stream) => LlmResponseStream::next(stream).await,
			Anthropic(stream) => LlmResponseStream::next(stream).await,
			Google(stream) => LlmResponseStream::next(stream).await,
			XAi(stream) => LlmResponseStream::next(stream).await,
			Mistral(stream) => LlmResponseStream::next(stream).await,
			PublicAi(stream) => LlmResponseStream::next(stream).await,
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

impl From<google::ResponseStream> for ResponseStream {
	fn from(stream: google::ResponseStream) -> Self {
		Self {
			inner: RespStreamInner::Google(stream),
		}
	}
}

impl From<xai::ResponseStream> for ResponseStream {
	fn from(stream: xai::ResponseStream) -> Self {
		Self {
			inner: RespStreamInner::XAi(stream),
		}
	}
}

impl From<mistral::ResponseStream> for ResponseStream {
	fn from(stream: mistral::ResponseStream) -> Self {
		Self {
			inner: RespStreamInner::Mistral(stream),
		}
	}
}

impl From<publicai::ResponseStream> for ResponseStream {
	fn from(stream: publicai::ResponseStream) -> Self {
		Self {
			inner: RespStreamInner::PublicAi(stream),
		}
	}
}
