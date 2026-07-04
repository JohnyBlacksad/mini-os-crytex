use std::collections::HashSet;

use crate::compress::ContentCompressor;
use crate::dedup::{jaccard_similarity, word_set};

/// Configuration for [`TextCompressor`].
#[derive(Debug, Clone)]
pub struct TextCompressorConfig {
    /// Target fraction of original characters to keep.
    pub target_ratio: f64,
    /// Minimum segments before crushing.
    pub min_segments: usize,
    /// Near-duplicate threshold (0..1) based on shared words.
    pub dedup_threshold: f64,
}

impl Default for TextCompressorConfig {
    fn default() -> Self {
        Self {
            target_ratio: 0.5,
            min_segments: 6,
            dedup_threshold: 0.85,
        }
    }
}

/// Extractive text compressor for large plain-text blocks.
#[derive(Debug, Clone, Default)]
pub struct TextCompressor {
    config: TextCompressorConfig,
}

impl TextCompressor {
    pub fn new(config: TextCompressorConfig) -> Self {
        Self { config }
    }
}

impl ContentCompressor for TextCompressor {
    fn compress(&self, content: &str, query: Option<&str>, _budget: usize) -> String {
        let segments = split_segments(content);
        if segments.len() < self.config.min_segments {
            return content.to_string();
        }

        let query_words: HashSet<String> = query
            .map(|q| {
                q.split_whitespace()
                    .map(|w| w.to_ascii_lowercase())
                    .filter(|w| w.len() > 2)
                    .collect()
            })
            .unwrap_or_default();

        let total_chars: usize = segments.iter().map(|s| s.len()).sum();
        let target_chars = (total_chars as f64 * self.config.target_ratio).max(100.0) as usize;

        let mut scored: Vec<(usize, f64, &str)> = segments
            .iter()
            .enumerate()
            .map(|(i, seg)| (i, score_segment(seg, &query_words), *seg))
            .collect();

        // Recency and position bonuses.
        let n = scored.len();
        for (i, score, _) in &mut scored {
            let position = (*i as f64 + 1.0) / n as f64;
            // Front and back are more important.
            *score += (1.0 - (position - 0.5).abs() * 2.0) * 0.3;
        }

        scored.sort_by(|a, b| crate::scoring::cmp_f64_desc(a.1, b.1));

        let mut kept = Vec::new();
        let mut kept_chars = 0usize;
        let mut word_sets: Vec<HashSet<String>> = Vec::new();

        for (idx, _, seg) in scored {
            if kept_chars >= target_chars {
                break;
            }
            let seg_words = word_set(seg);
            if word_sets
                .iter()
                .any(|s| jaccard_similarity(&seg_words, s) >= self.config.dedup_threshold)
            {
                continue;
            }
            kept.push((idx, seg));
            kept_chars += seg.len();
            word_sets.push(seg_words);
        }

        if kept.len() < 2 {
            return content.to_string();
        }

        // Restore original order.
        kept.sort_by_key(|a| a.0);
        kept.into_iter()
            .map(|(_, seg)| seg)
            .collect::<Vec<_>>()
            .join("\n")
    }
}

fn split_segments(text: &str) -> Vec<&str> {
    text.lines()
        .flat_map(|line| {
            // Split on sentence boundaries.
            let mut start = 0usize;
            let mut out = Vec::new();
            for (i, c) in line.char_indices() {
                if matches!(c, '.' | '!' | '?') {
                    let end = i + c.len_utf8();
                    let seg = &line[start..end].trim();
                    if !seg.is_empty() {
                        out.push(*seg);
                    }
                    start = end;
                }
            }
            let tail = line[start..].trim();
            if !tail.is_empty() {
                out.push(tail);
            }
            out
        })
        .collect()
}

fn score_segment(seg: &str, query_words: &HashSet<String>) -> f64 {
    let lower = seg.to_ascii_lowercase();
    let mut score = 0.0;
    for word in query_words {
        if lower.contains(word) {
            score += 1.0;
        }
    }
    if lower.contains("error") || lower.contains("fail") || lower.contains("panic") {
        score += 1.5;
    }
    if lower.contains("todo") || lower.contains("fixme") || lower.contains("important") {
        score += 0.5;
    }
    // Penalize very short segments.
    if seg.chars().count() < 12 {
        score -= 0.5;
    }
    score
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_text_passes_through() {
        let c = TextCompressor::default();
        let text = "Hello world.";
        assert_eq!(c.compress(text, None, 100), text);
    }

    #[test]
    fn long_text_compresses() {
        let c = TextCompressor::default();
        let mut text = String::new();
        for i in 0..50 {
            text.push_str(&format!(
                "This is sentence number {} describing routine work. ",
                i
            ));
        }
        text.push_str("CRITICAL: the system failed to respond.");
        let out = c.compress(&text, Some("failed"), 1000);
        assert!(out.len() < text.len());
        assert!(out.contains("CRITICAL") || out.contains("failed"));
    }
}
