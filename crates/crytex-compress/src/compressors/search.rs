use std::collections::BTreeMap;

use regex::Regex;

use crate::compress::ContentCompressor;

/// Configuration for [`SearchCompressor`].
#[derive(Debug, Clone)]
pub struct SearchCompressorConfig {
    /// Max matches kept per file.
    pub max_matches_per_file: usize,
    /// Max total matches across all files.
    pub max_total_matches: usize,
    /// Max files kept.
    pub max_files: usize,
    /// Minimum input lines before compression.
    pub min_lines: usize,
}

impl Default for SearchCompressorConfig {
    fn default() -> Self {
        Self {
            max_matches_per_file: 20,
            max_total_matches: 100,
            max_files: 20,
            min_lines: 20,
        }
    }
}

/// Content-aware compressor for grep/ripgrep output.
#[derive(Debug, Clone, Default)]
pub struct SearchCompressor {
    config: SearchCompressorConfig,
}

impl SearchCompressor {
    pub fn new(config: SearchCompressorConfig) -> Self {
        Self { config }
    }
}

impl ContentCompressor for SearchCompressor {
    fn compress(&self, content: &str, query: Option<&str>, _budget: usize) -> String {
        let lines: Vec<&str> = content.lines().collect();
        if lines.len() < self.config.min_lines {
            return content.to_string();
        }

        let query_words = query
            .map(|q| {
                q.split_whitespace()
                    .map(|w| w.to_ascii_lowercase())
                    .filter(|w| w.len() > 2)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        let re = Regex::new(r"^([^:]+):(\d+):(.*)$").unwrap();

        #[derive(Debug, Clone)]
        struct Match {
            file: String,
            line_no: usize,
            content: String,
            score: f64,
        }

        let mut matches: Vec<Match> = Vec::new();
        for line in &lines {
            if let Some(caps) = re.captures(line) {
                let file = caps[1].to_string();
                let line_no: usize = caps[2].parse().unwrap_or(0);
                let content = caps[3].to_string();
                let lower = content.to_ascii_lowercase();
                let mut score = 0.0;
                for word in &query_words {
                    if lower.contains(word) {
                        score += 1.0;
                    }
                }
                if lower.contains("error") || lower.contains("fail") || lower.contains("panic") {
                    score += 2.0;
                }
                matches.push(Match {
                    file,
                    line_no,
                    content,
                    score,
                });
            }
        }

        if matches.is_empty() {
            return content.to_string();
        }

        // Group by file.
        let mut by_file: BTreeMap<String, Vec<Match>> = BTreeMap::new();
        for m in matches {
            by_file.entry(m.file.clone()).or_default().push(m);
        }

        // Cap files.
        let mut files: Vec<(String, Vec<Match>)> = by_file.into_iter().collect();
        if files.len() > self.config.max_files {
            files.sort_by_key(|a| std::cmp::Reverse(a.1.len()));
            files.truncate(self.config.max_files);
        }

        let mut out = String::new();
        let mut total = 0usize;
        for (file, mut file_matches) in files {
            file_matches.sort_by(|a, b| crate::scoring::cmp_f64_desc(a.score, b.score));
            let keep = file_matches
                .into_iter()
                .take(self.config.max_matches_per_file)
                .filter(|_| {
                    if total < self.config.max_total_matches {
                        total += 1;
                        true
                    } else {
                        false
                    }
                })
                .collect::<Vec<_>>();
            if keep.is_empty() {
                continue;
            }
            out.push_str(&format!("\n{}\n", file));
            for m in keep {
                out.push_str(&format!("{}:{}:{}\n", m.file, m.line_no, m.content));
            }
        }

        if out.is_empty() {
            return content.to_string();
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_search_passes_through() {
        let c = SearchCompressor::default();
        let text = "src/main.rs:1: fn main() {}\n";
        assert_eq!(c.compress(text, None, 100), text);
    }

    #[test]
    fn long_search_compresses() {
        let c = SearchCompressor::default();
        let mut text = String::new();
        for i in 0..200 {
            text.push_str(&format!("src/file{}.rs:{}: println!()\n", i % 10, i));
        }
        let out = c.compress(&text, Some("println"), 1000);
        assert!(out.lines().count() < text.lines().count());
    }
}
