use std::fs;

use serde::Deserialize;
use serde_json::json;
use soe_llms::{
	Input, Llms, LlmsConfig, Model, Output, Request, ResponseEvent, Role, Tool,
};

#[derive(Debug, Clone, Deserialize)]
struct ToolInput {
	name: String,
}

#[tokio::main]
async fn main() {
	tracing_subscriber::fmt()
		.with_env_filter("soe_llms=info,warn")
		.init();

	let llms = Llms::new(
		LlmsConfig::new()
			.openai(
				fs::read_to_string("../.env.openai")
					.unwrap()
					.trim()
					.to_string(),
			)
			.anthropic(
				fs::read_to_string("../.env.anthropic")
					.unwrap()
					.trim()
					.to_string(),
			),
	);

	let mut req = Request {
		input: vec![],
		instructions: "You are a helpful assistant.".into(),
		model: Model::ClaudeHaiku4_5,
		user_id: "example_script".into(),
		tools: vec![Tool {
			name: "test_toolcall".into(),
			description: "A test toolcall for demonstration purposes.".into(),
			parameters: Some(json!({
				"type": "object",
				"properties": {
					"name": {
						"type": "string",
						"description": "A name to analyze.",
					}
				},
				"required": ["name"],
			})),
		}],
	};

	req.input = vec![Input::Text {
		role: Role::User,
		content:
			"First tell me a random name you choose, and why you chose it. \
			Then call the test function, and afterwards answer the question \
			the tool responded with."
				.into(),
	}];

	let mut stream = llms.request(&req).await.unwrap();

	let mut response = None;
	while let Some(thing) = stream.next().await {
		match thing.unwrap() {
			ResponseEvent::TextDelta { content } => eprint!("{}", content),
			ResponseEvent::Completed(resp) => response = Some(resp),
		}
	}
	eprint!("\n");

	let response = response.unwrap();
	eprintln!("Final output: {:?}", response);

	for output in response.output {
		match output {
			Output::ToolCall { name, input, id } => {
				assert_eq!(name, "test_toolcall");

				req.input.push(Input::ToolCall {
					id: id.clone(),
					name,
					input: input.clone(),
				});

				let input: ToolInput = serde_json::from_value(input).unwrap();

				req.input.push(Input::ToolCallOutput {
					id,
					output: format!(
						"What is the meaning of the name '{}'",
						input.name
					),
				});
			}
			o => req.input.push(o.into()),
		}
	}

	let mut stream = llms.request(&req).await.unwrap();

	let mut response = None;
	while let Some(thing) = stream.next().await {
		match thing.unwrap() {
			ResponseEvent::TextDelta { content } => eprint!("{}", content),
			ResponseEvent::Completed(resp) => response = Some(resp),
		}
	}
	eprint!("\n");

	let response = response.unwrap();
	eprintln!("Final output: {:?}", response);

	req.input
		.extend(response.output.into_iter().map(Into::into));

	eprintln!("{:#?}", req.input);
}
