//! Smoke-test every supported model with a single trivial query.
//!
//! Models whose API key is missing are skipped. Run with:
//! `cargo run --example all`.

use std::fs;

use soe_llms::{
	Input, Llms, LlmsConfig, LlmsError, Model, Request, ResponseEvent, Role,
};

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
			.publicai(read_env("../.env.publicai")),
	);

	for &model in Model::ALL {
		eprintln!("\n=== {model:?} ===");

		match run_model(&llms, model).await {
			Ok(()) => {}
			Err(LlmsError::LlmNotConfigured(provider)) => {
				eprintln!("skipped: {provider} API key not configured");
			}
			Err(e) => {
				eprintln!("FAILED: {e}");
			}
		}
	}
}

async fn run_model(llms: &Llms, model: Model) -> Result<(), LlmsError> {
	let req = Request {
		input: vec![Input::Text {
			role: Role::User,
			content: "Pick a random color and reply with just its name.".into(),
		}],
		instructions: "You are a helpful assistant.".into(),
		model,
		user_id: "example_all".into(),
		tools: vec![],
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
