use std::io;

use reqwest::StatusCode;

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum LlmsError {
	#[error("Llm not configured: {0}")]
	LlmNotConfigured(String),
	#[error("JSON deserialization error: {0}")]
	Json(#[from] serde_json::Error),
	#[error("Response error: status {status}, body {body}")]
	Response { status: StatusCode, body: String },
	#[error("Reqwest error: {0}")]
	Reqwest(#[from] reqwest::Error),
	#[error("IO error: {0}")]
	Io(#[from] io::Error),
}
