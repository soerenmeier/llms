use reqwest::StatusCode;

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum LlmsError {
	#[error("Response error: status {status}, body {body}")]
	ResponseError { status: StatusCode, body: String },
	#[error("Reqwest error: {0}")]
	ReqwestError(#[from] reqwest::Error),
}
