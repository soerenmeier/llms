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

struct LlmsProvider {
	open_ai: Option<openai::OpenAi>,
}

pub struct Llms {
	inner: LlmsProvider,
}

impl Llms {
	pub fn new(config: LlmsConfig) -> Self {
		Self {
			inner: LlmsProvider {
				open_ai: config.openai_api_key.map(openai::OpenAi::new),
			},
		}
	}

	pub async fn request(
		&self,
		req: &Request,
	) -> Result<Box<dyn ResponseStream>, LlmsError> {
		todo!()
	}
}

// pub trait Llms {
// 	async fn request()
// }
