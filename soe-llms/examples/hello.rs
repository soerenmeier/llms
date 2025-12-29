use std::fs;

use soe_llms::openai::{
	CustomToolCallOutput, Input, InputMessage, OpenAi, OpenAiModel, Request,
	Role, Tool,
};

#[tokio::main]
async fn main() {
	let open_ai = OpenAi::new(
		fs::read_to_string("../.env.openai")
			.unwrap()
			.trim()
			.to_string(),
	);

	let mut req = Request {
		input: vec![],
		instructions: "You are a helpful assistant.".into(),
		model: OpenAiModel::Gpt5,
		prompt_cache_key: "example_script".into(),
		safety_identifier: "example_script".into(),
		tools: vec![Tool::Custom {
			name: "test_toolcall".into(),
			description: "A test toolcall for demonstration purposes.".into(),
		}],
	};

	req.input = vec![Input::Message(InputMessage::Input {
					role: Role::User,
					content: "Please call test toolcall with a random name and then tell what that name has as meaning.".into(),
				})];

	let mut stream = open_ai.request(&req).await.unwrap();

	while let Some(thing) = stream.next().await {
		println!("{:?}", thing);
	}

	let output = stream.completed_response.take().unwrap();
	eprintln!("Final output: {:#?}", output);

	req.input = output.output.into_iter().map(|o| o.into()).collect();

	{
		match req.input.last().unwrap() {
			Input::CustomToolCall(ct) => {
				assert_eq!(ct.name, "test_toolcall");

				req.input.push(Input::CustomToolCallOutput(
					CustomToolCallOutput {
						id: None,
						call_id: ct.call_id.clone(),
						output: format!(
							"The name you chose was '{}'",
							ct.input
						),
					},
				));
			}
			_ => panic!("unexpected, not a toolcall"),
		}
	}

	// the last input should be a toolcall
	let mut stream = open_ai.request(&req).await.unwrap();

	while let Some(thing) = stream.next().await {
		println!("{:?}", thing);
	}

	let output = stream.completed_response.take().unwrap();
	eprintln!("Final output: {:#?}", output);
}
