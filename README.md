# LLMS

Crates for working with large language models (LLMs) in Rust.

These are opiniated implementations and are intended for agent use.

## soe-llms

Provides a common interface for working with different providers.
Currently the following models are supported:

```rust
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum Model {
	Gpt6Astra,
	Gpt6_1Sol,
	Gpt6Luna,

	ClaudeFable5_1,
	ClaudeOpus5_5,
	ClaudeSonnet5_5,
	ClaudeHaiku5_5,

	Gemini4Argon,
	GeminiFlash3_8,
	GeminiFlash3_5Lite,

	Grok4_7,

	MistralLarge,
	MistralMedium,
	MistralSmall,
	Ministral14b,

	// At the moment tool calls are not supported
	Apertus8bInstruct,

	OpenRouter(Option<String>),
}
```
