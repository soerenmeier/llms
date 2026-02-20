# soe-llms

Crate for working with large language models (LLMs) in Rust.

These are opiniated implementations and are intended for agent use
no image support is planned.

Provides a common interface for working with different providers.
Currently the following models are supported:

```rust
pub enum Model {
	Gpt5,
	Gpt5Mini,
	Gpt5Nano,
	Gpt5_2,
	ClaudeOpus4_6,
	ClaudeSonnet4_6,
	ClaudeHaiku4_5,
	GeminiPro3,
	GeminiFlash3,
	Grok4_1Fast,
	Grok4_1FastNonReasoning,
	GrokCodeFast1,
	MistralLarge3,
	MistralMedium3_1,
	MistralSmall3_2,
	Devstral2,
	MagistralMedium1_2,
	// At the moment tool calls are not supported
	Apertus8bInstruct,
}
```
