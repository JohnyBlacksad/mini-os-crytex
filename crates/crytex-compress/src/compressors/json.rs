use std::collections::HashSet;

use serde_json::{Map, Value};

use crate::compress::ContentCompressor;

use super::smart_crusher::{SmartCrusher, SmartCrusherConfig};

/// Configuration for [`JsonCompressor`].
#[derive(Debug, Clone)]
pub struct JsonCompressorConfig {
    /// Max items kept from a JSON array of objects.
    pub max_array_items: usize,
    /// Max string length before truncation.
    pub max_string_len: usize,
    /// Min input chars before compression.
    pub min_chars: usize,
    /// Whether to minify non-array JSON.
    pub minify: bool,
}

impl Default for JsonCompressorConfig {
    fn default() -> Self {
        Self {
            max_array_items: 50,
            max_string_len: 500,
            min_chars: 200,
            minify: true,
        }
    }
}

/// Content-aware compressor for JSON payloads.
#[derive(Debug, Clone, Default)]
pub struct JsonCompressor {
    config: JsonCompressorConfig,
}

impl JsonCompressor {
    pub fn new(config: JsonCompressorConfig) -> Self {
        Self { config }
    }
}

impl ContentCompressor for JsonCompressor {
    fn compress(&self, content: &str, query: Option<&str>, _budget: usize) -> String {
        if content.len() < self.config.min_chars {
            return content.to_string();
        }

        let value: Value = match serde_json::from_str(content) {
            Ok(v) => v,
            Err(_) => return content.to_string(),
        };

        let query_words = query
            .map(|q| {
                q.split_whitespace()
                    .map(|w| w.to_ascii_lowercase())
                    .filter(|w| w.len() > 2)
                    .collect::<HashSet<_>>()
            })
            .unwrap_or_default();

        match value {
            Value::Array(arr) if !arr.is_empty() && arr.iter().all(|v| v.is_object()) => {
                if arr.len() <= self.config.max_array_items {
                    compress_object_array(arr, &self.config, &query_words)
                } else {
                    let crusher = SmartCrusher::new(SmartCrusherConfig {
                        max_samples: self.config.max_array_items,
                        max_unique_values: self.config.max_array_items,
                        min_chars: self.config.min_chars,
                        max_string_len: self.config.max_string_len,
                    });
                    crusher.compress(content, query, _budget)
                }
            }
            other => {
                if self.config.minify {
                    serde_json::to_string(&other).unwrap_or_else(|_| content.to_string())
                } else {
                    content.to_string()
                }
            }
        }
    }
}

fn compress_object_array(
    arr: Vec<Value>,
    config: &JsonCompressorConfig,
    query_words: &HashSet<String>,
) -> String {
    if arr.len() <= config.max_array_items {
        let trimmed: Vec<Value> = arr
            .into_iter()
            .map(|v| trim_value(v, config.max_string_len))
            .collect();
        return serde_json::to_string(&trimmed).unwrap_or_default();
    }

    // Score each item by query overlap and schema richness.
    let mut scored: Vec<(usize, f64, Value)> = arr
        .into_iter()
        .enumerate()
        .map(|(idx, v)| {
            let score = score_json_value(&v, query_words);
            (idx, score, v)
        })
        .collect();

    // Keep first and last, plus top-scored middle items.
    scored.sort_by(|a, b| crate::scoring::cmp_f64_desc(a.1, b.1));
    let mut keep = HashSet::new();
    keep.insert(0);
    if !scored.is_empty() {
        keep.insert(scored.len() - 1);
    }
    for (idx, _, _) in scored.iter().take(config.max_array_items) {
        keep.insert(*idx);
    }

    let mut selected: Vec<(usize, Value)> = scored
        .into_iter()
        .filter(|(idx, _, _)| keep.contains(idx))
        .map(|(idx, _, v)| (idx, trim_value(v, config.max_string_len)))
        .collect();

    // Restore original order.
    selected.sort_by_key(|a| a.0);
    let selected: Vec<Value> = selected.into_iter().map(|(_, v)| v).collect();

    serde_json::to_string(&selected).unwrap_or_default()
}

fn score_json_value(value: &Value, query_words: &HashSet<String>) -> f64 {
    let text = json_to_searchable_text(value).to_ascii_lowercase();
    let mut score = 0.0;
    for word in query_words {
        if text.contains(word) {
            score += 1.0;
        }
    }
    // Prefer items with more fields (richer schema sample).
    if let Value::Object(map) = value {
        score += (map.len() as f64 * 0.05).min(1.0);
    }
    score
}

fn json_to_searchable_text(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Array(arr) => arr
            .iter()
            .map(json_to_searchable_text)
            .collect::<Vec<_>>()
            .join(" "),
        Value::Object(map) => map
            .values()
            .map(json_to_searchable_text)
            .collect::<Vec<_>>()
            .join(" "),
        Value::Number(n) => n.to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Null => String::new(),
    }
}

fn trim_value(value: Value, max_len: usize) -> Value {
    match value {
        Value::String(s) => {
            if s.len() > max_len {
                Value::String(format!("{}… [truncated]", &s[..max_len]))
            } else {
                Value::String(s)
            }
        }
        Value::Array(arr) => {
            Value::Array(arr.into_iter().map(|v| trim_value(v, max_len)).collect())
        }
        Value::Object(map) => {
            let trimmed: Map<String, Value> = map
                .into_iter()
                .map(|(k, v)| (k, trim_value(v, max_len)))
                .collect();
            Value::Object(trimmed)
        }
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_json_passes_through() {
        let c = JsonCompressor::default();
        let text = r#"{"a": 1}"#;
        assert_eq!(c.compress(text, None, 100), text);
    }

    #[test]
    fn large_array_is_reduced() {
        let c = JsonCompressor::default();
        let arr: Vec<Value> = (0..200)
            .map(|i| {
                let mut obj = serde_json::Map::new();
                obj.insert("id".to_string(), Value::Number(i.into()));
                obj.insert("msg".to_string(), Value::String(format!("item {}", i)));
                Value::Object(obj)
            })
            .collect();
        let text = serde_json::to_string(&arr).unwrap();
        let out = c.compress(&text, Some("item 199"), 1000);
        let parsed: Value = serde_json::from_str(&out).unwrap();
        let samples = parsed.get("_samples").unwrap().as_array().unwrap();
        assert!(samples.len() < arr.len());
        // Query-matching row should be prioritized.
        assert!(samples.iter().any(|v| {
            v.get("msg")
                .and_then(|m| m.as_str())
                .map(|m| m == "item 199")
                .unwrap_or(false)
        }));
    }

    #[test]
    fn non_object_array_minified() {
        let c = JsonCompressor::new(JsonCompressorConfig {
            min_chars: 1,
            ..Default::default()
        });
        let text = "[ 1 , 2 , 3 ]";
        assert_eq!(c.compress(text, None, 100), "[1,2,3]");
    }
}
