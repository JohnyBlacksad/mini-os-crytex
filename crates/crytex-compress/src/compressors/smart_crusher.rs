use std::collections::{BTreeMap, HashMap, HashSet};

use serde_json::{Map, Value};

use crate::compress::ContentCompressor;

/// Configuration for [`SmartCrusher`].
#[derive(Debug, Clone)]
pub struct SmartCrusherConfig {
    /// Max number of representative rows to keep.
    pub max_samples: usize,
    /// Max unique values to report per key in the summary.
    pub max_unique_values: usize,
    /// Minimum input chars before crushing kicks in.
    pub min_chars: usize,
    /// Max string length inside samples.
    pub max_string_len: usize,
}

impl Default for SmartCrusherConfig {
    fn default() -> Self {
        Self {
            max_samples: 50,
            max_unique_values: 20,
            min_chars: 200,
            max_string_len: 500,
        }
    }
}

/// Headroom-style smart crusher for JSON arrays of objects.
///
/// Infers the schema, deduplicates rows, aggregates per-field statistics, and
/// keeps a representative sample set.
#[derive(Debug, Clone, Default)]
pub struct SmartCrusher {
    config: SmartCrusherConfig,
}

impl SmartCrusher {
    pub fn new(config: SmartCrusherConfig) -> Self {
        Self { config }
    }
}

impl ContentCompressor for SmartCrusher {
    fn compress(&self, content: &str, query: Option<&str>, _budget: usize) -> String {
        if content.len() < self.config.min_chars {
            return content.to_string();
        }

        let value: Vec<Value> = match serde_json::from_str(content) {
            Ok(Value::Array(arr)) if !arr.is_empty() && arr.iter().all(Value::is_object) => arr,
            Ok(Value::Array(_)) => {
                // Non-object array: just minify.
                return serde_json::to_string(&serde_json::from_str::<Value>(content).unwrap())
                    .unwrap_or_else(|_| content.to_string());
            }
            _ => return content.to_string(),
        };

        let query_words: HashSet<String> = query
            .map(|q| {
                q.split_whitespace()
                    .map(|w| w.to_ascii_lowercase())
                    .filter(|w| w.len() > 2)
                    .collect()
            })
            .unwrap_or_default();

        let result = crush_array(value, &self.config, &query_words);
        serde_json::to_string(&result).unwrap_or_else(|_| content.to_string())
    }
}

fn crush_array(
    rows: Vec<Value>,
    config: &SmartCrusherConfig,
    query_words: &HashSet<String>,
) -> Value {
    let total_rows = rows.len();

    // Schema and stats.
    let mut key_stats: BTreeMap<String, KeyStats> = BTreeMap::new();
    for row in &rows {
        if let Value::Object(map) = row {
            for (k, v) in map {
                key_stats
                    .entry(k.clone())
                    .or_default()
                    .add(v, config.max_unique_values);
            }
        }
    }

    // Deduplicate rows by canonical JSON.
    let mut seen: HashSet<String> = HashSet::new();
    let mut unique_rows: Vec<(usize, Value)> = Vec::new();
    for (idx, row) in rows.into_iter().enumerate() {
        let canonical = canonical_json(&row);
        if seen.insert(canonical) {
            unique_rows.push((idx, row));
        }
    }

    // Score and select representative rows.
    let mut scored: Vec<(usize, f64, Value)> = unique_rows
        .into_iter()
        .map(|(idx, row)| {
            let score = score_row(&row, query_words, &key_stats);
            (idx, score, row)
        })
        .collect();

    scored.sort_by(|a, b| crate::scoring::cmp_f64_desc(a.1, b.1));

    let mut keep = HashSet::new();
    keep.insert(0);
    if total_rows > 1 {
        keep.insert(total_rows - 1);
    }

    // Ensure coverage of important categorical keys.
    let categorical: Vec<String> = key_stats
        .iter()
        .filter(|(_, s)| s.unique_count() > 1 && s.unique_count() <= 20)
        .map(|(k, _)| k.clone())
        .collect();

    let mut covered: HashMap<String, HashSet<String>> = HashMap::new();
    for (idx, _, row) in &scored {
        if keep.len() >= config.max_samples {
            break;
        }
        let mut useful = false;
        if let Value::Object(map) = row {
            for key in &categorical {
                if let Some(v) = map.get(key) {
                    let value_str = json_summary_value(v);
                    let set = covered.entry(key.clone()).or_default();
                    if set.insert(value_str.clone()) {
                        useful = true;
                    }
                }
            }
        }
        if useful || keep.len() < config.max_samples / 2 {
            keep.insert(*idx);
        }
    }

    // Fill remaining slots with top-scored rows.
    for (idx, _, _) in &scored {
        if keep.len() >= config.max_samples {
            break;
        }
        keep.insert(*idx);
    }

    let mut selected: Vec<(usize, Value)> = scored
        .into_iter()
        .filter(|(idx, _, _)| keep.contains(idx))
        .map(|(idx, _, v)| (idx, trim_value(v, config.max_string_len)))
        .collect();
    selected.sort_by_key(|a| a.0);
    let samples: Vec<Value> = selected.into_iter().map(|(_, v)| v).collect();

    let mut summary = Map::new();
    summary.insert("total_rows".to_string(), Value::Number(total_rows.into()));
    summary.insert("unique_rows".to_string(), Value::Number(keep.len().into()));
    summary.insert(
        "fields".to_string(),
        Value::Array(key_stats.keys().map(|k| Value::String(k.clone())).collect()),
    );

    let mut field_stats = Map::new();
    for (key, stats) in key_stats {
        field_stats.insert(key, stats.to_value(config.max_unique_values));
    }
    summary.insert("field_stats".to_string(), Value::Object(field_stats));

    let mut output = Map::new();
    output.insert("_smart_crusher".to_string(), Value::Object(summary));
    output.insert("_samples".to_string(), Value::Array(samples));
    Value::Object(output)
}

#[derive(Default)]
struct KeyStats {
    count: usize,
    unique: HashSet<String>,
    numeric_min: Option<f64>,
    numeric_max: Option<f64>,
}

impl KeyStats {
    fn add(&mut self, value: &Value, max_unique: usize) {
        self.count += 1;
        let text = json_summary_value(value);
        if self.unique.len() < max_unique {
            self.unique.insert(text);
        }
        if let Some(n) = value.as_f64() {
            self.numeric_min = Some(self.numeric_min.map_or(n, |m| m.min(n)));
            self.numeric_max = Some(self.numeric_max.map_or(n, |m| m.max(n)));
        }
    }

    fn unique_count(&self) -> usize {
        self.unique.len()
    }

    fn to_value(&self, max_unique: usize) -> Value {
        let mut map = Map::new();
        map.insert("count".to_string(), Value::Number(self.count.into()));
        map.insert(
            "unique_count".to_string(),
            Value::Number(self.unique_count().into()),
        );
        let values: Vec<Value> = self
            .unique
            .iter()
            .take(max_unique)
            .map(|s| Value::String(s.clone()))
            .collect();
        map.insert("unique_values".to_string(), Value::Array(values));
        if let (Some(min), Some(max)) = (self.numeric_min, self.numeric_max) {
            map.insert("min".to_string(), json_number(min));
            map.insert("max".to_string(), json_number(max));
        }
        Value::Object(map)
    }
}

fn json_number(n: f64) -> Value {
    if n.fract() == 0.0 && n.is_finite() {
        Value::Number((n as i64).into())
    } else {
        serde_json::Number::from_f64(n)
            .map(Value::Number)
            .unwrap_or(Value::Null)
    }
}

fn json_summary_value(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Number(n) => n.to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Null => "null".to_string(),
        other => serde_json::to_string(other).unwrap_or_default(),
    }
}

fn canonical_json(value: &Value) -> String {
    match value {
        Value::Object(map) => {
            let sorted: BTreeMap<String, Value> = map.clone().into_iter().collect();
            serde_json::to_string(&sorted).unwrap_or_default()
        }
        _ => serde_json::to_string(value).unwrap_or_default(),
    }
}

fn score_row(
    row: &Value,
    query_words: &HashSet<String>,
    stats: &BTreeMap<String, KeyStats>,
) -> f64 {
    let mut score = 0.0;
    if let Value::Object(map) = row {
        let text = json_to_searchable_text(row).to_ascii_lowercase();
        for word in query_words {
            if text.contains(word) {
                score += 2.0;
            }
        }
        // Prefer rows with rare categorical values (more information).
        for (key, value) in map {
            if let Some(s) = stats.get(key)
                && s.unique_count() > 1
                && s.unique_count() <= 20
            {
                let val = json_summary_value(value);
                // Rarer value -> higher score.
                if s.unique.contains(&val) {
                    score += 0.1;
                }
            }
        }
        // Prefer rows with more populated fields.
        score += (map.len() as f64 * 0.02).min(0.5);
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
        let c = SmartCrusher::default();
        let text = r#"[{"a": 1}]"#;
        assert_eq!(c.compress(text, None, 100), text);
    }

    #[test]
    fn large_array_crushed() {
        let c = SmartCrusher::default();
        let rows: Vec<Value> = (0..200)
            .map(|i| {
                let mut obj = Map::new();
                obj.insert("id".to_string(), Value::Number(i.into()));
                obj.insert(
                    "status".to_string(),
                    Value::String(if i % 2 == 0 {
                        "ok".into()
                    } else {
                        "fail".into()
                    }),
                );
                obj.insert("msg".to_string(), Value::String(format!("message {}", i)));
                Value::Object(obj)
            })
            .collect();
        let text = serde_json::to_string(&rows).unwrap();
        let out = c.compress(&text, Some("fail"), 1000);
        let parsed: Value = serde_json::from_str(&out).unwrap();
        assert!(parsed.get("_smart_crusher").is_some());
        assert!(parsed.get("_samples").is_some());
        let samples = parsed.get("_samples").unwrap().as_array().unwrap();
        assert!(samples.len() < rows.len());
    }

    #[test]
    fn deduplicates_identical_rows() {
        let c = SmartCrusher::new(SmartCrusherConfig {
            min_chars: 1,
            ..Default::default()
        });
        let rows = vec![
            serde_json::json!({"status": "ok"}),
            serde_json::json!({"status": "ok"}),
            serde_json::json!({"status": "ok"}),
        ];
        let text = serde_json::to_string(&rows).unwrap();
        let out = c.compress(&text, None, 1000);
        let parsed: Value = serde_json::from_str(&out).unwrap();
        let total = parsed["_smart_crusher"]["total_rows"].as_u64().unwrap();
        assert_eq!(total, 3);
    }
}
