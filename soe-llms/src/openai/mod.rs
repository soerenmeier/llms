use eventsource_stream::Eventsource;
use futures::StreamExt;
use reqwest::{
	Client,
	header::{ACCEPT, CONTENT_TYPE, HeaderValue},
};
use serde::{Deserialize, Serialize};

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
		req: Request,
	) -> Result<Vec<Output>, OpenAiError> {
		#[derive(Debug, Serialize, Deserialize)]
		struct Req {
			input: Vec<Input>,
			instructions: String,
			model: OpenAiModel,
			prompt_cache_key: String,
			safety_identifier: String,
			tools: Vec<Tool>,
			stream: bool,
		}

		let req = Req {
			input: req.input,
			instructions: req.instructions,
			model: req.model,
			prompt_cache_key: req.prompt_cache_key,
			safety_identifier: req.safety_identifier,
			tools: req.tools,
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

		let client_error = resp.status().is_client_error();
		if !resp.status().is_success() {
			let text = resp.text().await?;

			if client_error {
				return Err(OpenAiError::ClientError(text));
			} else {
				return Err(OpenAiError::OtherError(text));
			}
		}

		eprintln!("resp {resp:?}");

		let mut stream = resp.bytes_stream().eventsource();

		while let Some(thing) = stream.next().await {
			println!("{:?}", thing);
		}

		todo!()
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
#[serde(rename_all = "lowercase", tag = "type")]
pub enum Tool {
	Custom { name: String, description: String },
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase", tag = "type")]
pub enum Input {
	Message { role: Role, content: String },
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
	Developer,
	User,
	Assistant,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum Output {}

#[derive(Debug, thiserror::Error)]
pub enum OpenAiError {
	#[error("Client error: {0}")]
	ClientError(String),
	#[error("Other error: {0}")]
	OtherError(String),
	#[error("Reqwest error: {0}")]
	ReqwestError(#[from] reqwest::Error),
}

#[derive(Debug, Serialize, Deserialize)]
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
