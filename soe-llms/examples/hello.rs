use std::fs;

use soe_llms::{
	Input, Llms, LlmsConfig, Model, Output, Request, ResponseEvent, Role, Tool,
};

#[tokio::main]
async fn main() {
	tracing_subscriber::fmt()
		.with_env_filter("soe_llms=info,warn")
		.init();

	let llms = Llms::new(LlmsConfig {
		openai_api_key: Some(
			fs::read_to_string("../.env.openai")
				.unwrap()
				.trim()
				.to_string(),
		),
	});

	let mut req = Request {
		input: vec![],
		instructions: "You are a helpful assistant.".into(),
		model: Model::Gpt5Nano,
		user_id: "example_script".into(),
		tools: vec![Tool {
			name: "test_toolcall".into(),
			description: "A test toolcall for demonstration purposes.".into(),
		}],
	};

	req.input = vec![Input::Text {
		role: Role::User,
		content: "Please call test toolcall with a random name \
						and then tell me what meaning that name has."
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

				req.input.push(Input::ToolCallOutput {
					id,
					output: format!("The name you chose was '{}'", input),
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
