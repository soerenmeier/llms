use std::{fmt, io};

use bytes::Bytes;
use futures::{StreamExt as _, TryStreamExt as _, stream::BoxStream};
use reqwest::Response;
use serde::de::DeserializeOwned;
use tokio::io::AsyncBufReadExt;
use tokio_util::io::StreamReader;

use crate::llms::LlmsError;

pub struct SseResponse {
	inner: StreamReader<BoxStream<'static, Result<Bytes, io::Error>>, Bytes>,
	line: Option<String>,
}

impl SseResponse {
	pub fn new(resp: Response) -> Self {
		assert!(
			resp.status().is_success(),
			"response is expected to be successful"
		);

		Self {
			inner: StreamReader::new(
				resp.bytes_stream()
					.map_err(|e| io::Error::new(io::ErrorKind::Other, e))
					.boxed(),
			),
			line: Some(String::with_capacity(1024)),
		}
	}

	pub async fn next<T: DeserializeOwned>(
		&mut self,
	) -> Option<Result<T, SseError>> {
		let line = self.line.as_mut()?;
		line.clear();

		let line = loop {
			match self.inner.read_line(line).await {
				Ok(0) => {
					self.line = None;
					return None;
				}
				Ok(_) => {}
				// get original error back
				Err(e) if e.kind() == io::ErrorKind::Other => {
					let err = match e.downcast::<reqwest::Error>() {
						Ok(e) => e.into(),
						Err(e) => e.into(),
					};

					return Some(Err(err));
				}
				Err(err) => return Some(Err(err.into())),
			}

			if let Some(line) = line.strip_prefix("data:") {
				break line.trim();
			}

			line.clear();
		};

		if line == "[DONE]" {
			return None;
		}

		Some(serde_json::from_str(line).map_err(|e| {
			eprintln!("received line {line}");
			e.into()
		}))
	}
}

impl fmt::Debug for SseResponse {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		f.debug_struct("SseResponse").finish()
	}
}

#[derive(Debug, thiserror::Error)]
pub enum SseError {
	#[error("IO error: {0}")]
	Io(#[from] io::Error),
	#[error("Reqwest error: {0}")]
	Reqwest(#[from] reqwest::Error),
	#[error("JSON deserialization error: {0}")]
	Json(#[from] serde_json::Error),
}

impl From<SseError> for LlmsError {
	fn from(value: SseError) -> Self {
		match value {
			SseError::Io(e) => LlmsError::Io(e),
			SseError::Reqwest(e) => LlmsError::Reqwest(e),
			SseError::Json(e) => LlmsError::Json(e),
		}
	}
}
