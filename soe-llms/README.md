# soe-llms

Crate for working with large language models (LLMs) in Rust.

These opinionated implementations are intended for agent use and provide a
common interface for different providers. Currently the following models are
supported:

```rust
pub enum Model {
    Gpt6Astra,
    Gpt6Sol,
    Gpt6Luna,

    ClaudeFable5_1,
    ClaudeOpus5_5,
    ClaudeSonnet5,
    ClaudeHaiku4_5,

    GeminiPro3_1,
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
