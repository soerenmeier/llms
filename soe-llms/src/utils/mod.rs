pub mod sse;

pub fn default_parameters() -> serde_json::Value {
	serde_json::json!({
		"type": "object",
		"properties": {},
	})
}
