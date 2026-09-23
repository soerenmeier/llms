//! Smoke-test image input on every model with known image support.
//!
//! Models whose API key is missing are skipped. Uses the JPEG in this folder.
//! Run with `cargo run --example all_images` (requests may incur API charges).

use std::fs;

use soe_llms::{
	ContentPart, Input, Llms, LlmsConfig, LlmsError, Model, Request,
	ResponseEvent, Role,
};

const IMAGE: &[u8] = include_bytes!("marcin-kempa-FXBZEsSK_MU-unsplash.jpg");

fn read_env(path: &str) -> Option<String> {
	fs::read_to_string(path).ok().map(|s| s.trim().to_string())
}

#[tokio::main]
async fn main() {
	tracing_subscriber::fmt()
		.with_env_filter("soe_llms=info,warn")
		.init();

	let llms = Llms::new(
		LlmsConfig::new()
			.openai(read_env("../.env.openai"))
			.anthropic(read_env("../.env.anthropic"))
			.google(read_env("../.env.google"))
			.xai(read_env("../.env.xai"))
			.mistral(read_env("../.env.mistral"))
			.openrouter(read_env("../.env.openrouter")),
	);

	for model in Model::ALL_IMAGES {
		eprintln!("\n=== {model:?} ===");

		match run_model(&llms, model.clone(), IMAGE).await {
			Ok(()) => {}
			Err(LlmsError::LlmNotConfigured(provider)) => {
				eprintln!("skipped: {provider} API key not configured");
			}
			Err(e) => eprintln!("FAILED: {e}"),
		}
	}
}

async fn run_model(
	llms: &Llms,
	model: Model,
	image: &[u8],
) -> Result<(), LlmsError> {
	let req = Request {
		input: vec![Input::Message {
			role: Role::User,
			parts: vec![
				ContentPart::Text {
					text: "Describe this image in one sentence.".into(),
				},
				ContentPart::Image {
					media_type: "image/jpeg".into(),
					data: image.to_vec(),
				},
			],
		}],
		instructions: "You are a helpful assistant.".into(),
		model,
		user_id: "example_all_images".into(),
		tools: vec![],
		reasoning_effort: None,
	};

	let mut stream = llms.request(&req).await?;

	while let Some(ev) = stream.next().await {
		match ev? {
			ResponseEvent::TextDelta { content } => eprint!("{content}"),
		}
	}
	eprintln!();

	Ok(())
}
