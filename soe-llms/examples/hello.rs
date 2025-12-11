use std::fs;

use soe_llms::openai::{Input, OpenAi, OpenAiModel, Request, Role, Tool};

#[tokio::main]
async fn main() {
	let open_ai = OpenAi::new(
		fs::read_to_string("../.env.openai")
			.unwrap()
			.trim()
			.to_string(),
	);

	let resp = open_ai
		.request(Request {
			input: vec![Input::Message {
				role: Role::User,
				content: "Please call test toolcall".into(),
			}],
			instructions: "You are a helpful assistant.".into(),
			model: OpenAiModel::Gpt5,
			prompt_cache_key: "example_script".into(),
			safety_identifier: "example_script".into(),
			tools: vec![Tool::Custom {
				name: "test_toolcall".into(),
				description: "A test toolcall for demonstration purposes."
					.into(),
			}],
		})
		.await
		.unwrap();

	eprintln!("{:#?}", resp);
}
