use std::collections::BTreeSet;

use unidiff::{Hunk, PatchSet, PatchedFile};

use crate::ccr::{CcrStore, compute_key};
use crate::compress::ContentCompressor;

/// Configuration for [`DiffCompressor`].
#[derive(Debug, Clone)]
pub struct DiffCompressorConfig {
    /// Context lines to keep on each side of a change.
    pub max_context_lines: usize,
    /// Max hunks kept per file.
    pub max_hunks_per_file: usize,
    /// Max files kept across the diff.
    pub max_files: usize,
    /// Minimum number of lines before compression is attempted.
    pub min_lines: usize,
    /// If true and a CCR store is provided, the original diff is stored and a
    /// retrieval marker is appended when compression is significant.
    pub enable_ccr: bool,
    /// Compression ratio below which CCR marker is emitted.
    pub ccr_ratio_threshold: f64,
}

impl Default for DiffCompressorConfig {
    fn default() -> Self {
        Self {
            max_context_lines: 2,
            max_hunks_per_file: 10,
            max_files: 20,
            min_lines: 20,
            enable_ccr: true,
            ccr_ratio_threshold: 0.8,
        }
    }
}

/// Content-aware compressor for unified diffs.
#[derive(Debug, Clone, Default)]
pub struct DiffCompressor {
    config: DiffCompressorConfig,
}

impl DiffCompressor {
    pub fn new(config: DiffCompressorConfig) -> Self {
        Self { config }
    }
}

impl ContentCompressor for DiffCompressor {
    fn compress(&self, content: &str, query: Option<&str>, _budget: usize) -> String {
        let lines: Vec<&str> = content.lines().collect();
        if lines.len() < self.config.min_lines {
            return content.to_string();
        }

        let mut patch = PatchSet::new();
        if patch.parse(content).is_err() || patch.is_empty() {
            return content.to_string();
        }

        let query_words = query.map(parse_query_words).unwrap_or_default();

        // Cap files by total changes, preserving original order.
        let mut files: Vec<&PatchedFile> = patch.files().iter().collect();
        let mut dropped_paths: BTreeSet<String> = BTreeSet::new();
        if files.len() > self.config.max_files {
            let mut indexed: Vec<(usize, usize)> = files
                .iter()
                .enumerate()
                .map(|(i, f)| (i, f.added() + f.removed()))
                .collect();
            indexed.sort_by_key(|a| std::cmp::Reverse(a.1));
            let mut keep = BTreeSet::new();
            for (i, _) in indexed.iter().take(self.config.max_files) {
                keep.insert(*i);
            }
            for (i, f) in files.iter().enumerate() {
                if !keep.contains(&i) {
                    dropped_paths.insert(f.path());
                }
            }
            files = files
                .into_iter()
                .enumerate()
                .filter(|(i, _)| keep.contains(i))
                .map(|(_, f)| f)
                .collect();
        }

        let mut output = String::new();
        for (idx, file) in files.iter().enumerate() {
            if idx > 0 {
                output.push('\n');
            }
            compress_file(file, &self.config, &query_words, &mut output);
        }

        if !dropped_paths.is_empty() {
            output.push_str("\n# ... diff compressor dropped ");
            output.push_str(&dropped_paths.len().to_string());
            output.push_str(" file(s): ");
            output.push_str(&dropped_paths.iter().cloned().collect::<Vec<_>>().join(", "));
            output.push('\n');
        }

        let out_lines = output.lines().count();
        if out_lines == 0 || out_lines > lines.len() {
            return content.to_string();
        }
        output
    }

    fn compress_with_store(
        &self,
        content: &str,
        query: Option<&str>,
        budget: usize,
        store: Option<&dyn CcrStore>,
    ) -> Result<String, crate::compress::CompressionError> {
        let compressed = self.compress(content, query, budget);
        if compressed == content {
            return Ok(compressed);
        }
        let Some(store) = store else {
            return Ok(compressed);
        };
        if !self.config.enable_ccr {
            return Ok(compressed);
        }
        let original_lines = content.lines().count();
        let compressed_lines = compressed.lines().count();
        if original_lines == 0 {
            return Ok(compressed);
        }
        let ratio = compressed_lines as f64 / original_lines as f64;
        if ratio >= self.config.ccr_ratio_threshold {
            return Ok(compressed);
        }
        let key = compute_key(content);
        store.put(&key, content.to_string())?;
        Ok(format!(
            "{}\n[original diff stored: ccr:{} (retrieve if needed)]",
            compressed, key
        ))
    }
}

fn parse_query_words(query: &str) -> Vec<String> {
    query
        .split_whitespace()
        .map(|w| w.to_ascii_lowercase())
        .filter(|w| w.len() > 2)
        .collect()
}

fn compress_file(
    file: &&PatchedFile,
    config: &DiffCompressorConfig,
    query_words: &[String],
    output: &mut String,
) {
    output.push_str(&format!("--- a/{}\n", file.path()));
    output.push_str(&format!("+++ b/{}\n", file.path()));

    let hunks: Vec<&Hunk> = file.hunks().iter().collect();
    let selected: Vec<&Hunk> = if hunks.len() > config.max_hunks_per_file {
        select_hunks(&hunks, config.max_hunks_per_file, query_words)
    } else {
        hunks
    };

    for hunk in selected {
        output.push_str(&format!(
            "@@ -{},{} +{},{} @@ {}\n",
            hunk.source_start,
            hunk.source_length,
            hunk.target_start,
            hunk.target_length,
            hunk.section_header
        ));
        let kept_lines = trim_context(hunk, config.max_context_lines);
        for line in kept_lines {
            output.push_str(&format!("{}{}\n", line.line_type, line.value));
        }
    }
}

fn select_hunks<'a>(hunks: &[&'a Hunk], max_hunks: usize, query_words: &[String]) -> Vec<&'a Hunk> {
    let mut scored: Vec<(usize, f64)> = hunks
        .iter()
        .enumerate()
        .map(|(idx, hunk)| (idx, score_hunk(hunk, query_words)))
        .collect();
    scored.sort_by(|a, b| crate::scoring::cmp_f64_desc(a.1, b.1));

    let mut keep = BTreeSet::new();
    keep.insert(0);
    keep.insert(hunks.len().saturating_sub(1));
    for (idx, _) in scored.iter().take(max_hunks) {
        keep.insert(*idx);
    }

    hunks
        .iter()
        .enumerate()
        .filter(|(idx, _)| keep.contains(idx))
        .map(|(_, h)| *h)
        .collect()
}

fn score_hunk(hunk: &Hunk, query_words: &[String]) -> f64 {
    let changes = hunk.added() + hunk.removed();
    let base = (changes as f64 * 0.03).min(0.3);
    let mut overlap = 0.0;
    if !query_words.is_empty() {
        let text = hunk
            .lines()
            .iter()
            .map(|l| l.value.to_ascii_lowercase())
            .collect::<Vec<_>>()
            .join(" ");
        for word in query_words {
            if text.contains(word) {
                overlap += 0.2;
            }
        }
    }
    (base + overlap).min(1.0)
}

fn trim_context(hunk: &Hunk, max_context: usize) -> Vec<unidiff::Line> {
    let lines = hunk.lines();
    if lines.is_empty() {
        return Vec::new();
    }
    let n = lines.len();
    let mut keep = vec![false; n];
    for (i, line) in lines.iter().enumerate() {
        if line.is_added() || line.is_removed() {
            let start = i.saturating_sub(max_context);
            let end = (i + max_context + 1).min(n);
            for slot in keep.iter_mut().take(end).skip(start) {
                *slot = true;
            }
        }
    }
    lines
        .iter()
        .enumerate()
        .filter(|(i, _)| keep[*i])
        .map(|(_, l)| l.clone())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_diff_passes_through() {
        let c = DiffCompressor::default();
        let text = "--- a/f\n+++ b/f\n@@ -1 +1 @@\n-old\n+new\n";
        assert_eq!(c.compress(text, None, 1000), text);
    }

    #[test]
    fn large_diff_compresses() {
        let c = DiffCompressor::default();
        let mut text = String::from("diff --git a/f.txt b/f.txt\n--- a/f.txt\n+++ b/f.txt\n");
        for i in 0..100 {
            text.push_str(&format!(
                "@@ -{0},5 +{0},5 @@\n context1\n context2\n-old{i}\n+new{i}\n context3\n context4\n",
                i + 1
            ));
        }
        let out = c.compress(&text, Some("new99"), 1000);
        assert!(out.lines().count() < text.lines().count());
    }

    #[test]
    fn ccr_store_receives_original() {
        use crate::ccr::InMemoryCcrStore;
        let c = DiffCompressor::default();
        let store = InMemoryCcrStore::new();
        let mut text = String::from("diff --git a/f.txt b/f.txt\n--- a/f.txt\n+++ b/f.txt\n");
        for i in 0..100 {
            text.push_str(&format!(
                "@@ -{0},5 +{0},5 @@\n context1\n context2\n-old{i}\n+new{i}\n context3\n context4\n",
                i + 1
            ));
        }
        let out = c
            .compress_with_store(&text, Some("new99"), 1000, Some(&store))
            .unwrap();
        assert!(out.contains("[original diff stored:"));
        let key_start = out.find("ccr:").unwrap() + 4;
        let key_end = out[key_start..]
            .find(' ')
            .map(|i| key_start + i)
            .unwrap_or(out.len());
        let key = &out[key_start..key_end];
        assert_eq!(store.get(key).unwrap(), Some(text));
    }
}
