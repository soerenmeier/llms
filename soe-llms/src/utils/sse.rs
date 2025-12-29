use std::io;

use bytes::Bytes;
use futures::{Stream, StreamExt as _, TryStreamExt as _, stream::BoxStream};
use reqwest::{Response, StatusCode, header::HeaderMap};
use serde::de::DeserializeOwned;
use tokio::io::AsyncBufReadExt;
use tokio_util::io::StreamReader;

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
				resp.bytes_stream().map_err(reqwest_to_io).boxed(),
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

		// eprintln!("received line {line}");

		Some(serde_json::from_str(line).map_err(|e| {
			eprintln!("received line {line}");
			e.into()
		}))
	}
}

#[derive(Debug, thiserror::Error)]
pub enum SseError {
	#[error("IO error: {0}")]
	Io(#[from] io::Error),
	#[error("JSON deserialization error: {0}")]
	Json(#[from] serde_json::Error),
}

fn reqwest_to_io(err: reqwest::Error) -> io::Error {
	io::Error::new(io::ErrorKind::Other, err)
}
