# LLMS

Crates for working with large language models (LLMs) in Rust.

These are opiniated implementations and are intended for agent use
no image support is planned.

## soe-llms

Provides a common interface for working with different providers.
Currently the following models are supported:

```rust
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum Model {
	Gpt5_6Sol,
	Gpt5_6Terra,
	Gpt5_6Luna,

	ClaudeFable5,
	ClaudeOpus4_8,
	ClaudeSonnet5,
	ClaudeHaiku4_5,

	GeminiPro3_1,
	GeminiFlash3_5,
	GeminiFlash3_1Lite,

	Grok4_5,

	MistralLarge,
	MistralMedium,
	MistralSmall,
	Ministral14b,

	// At the moment tool calls are not supported
	Apertus8bInstruct,

	OpenRouter(Option<String>),
}
```
