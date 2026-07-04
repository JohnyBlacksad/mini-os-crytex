use crate::compress::ContentCompressor;

/// Configuration for [`LogCompressor`].
#[derive(Debug, Clone)]
pub struct LogCompressorConfig {
    /// Max total output lines.
    pub max_lines: usize,
    /// Lines of context around error/fail lines.
    pub context_lines: usize,
    /// Minimum input lines before compression kicks in.
    pub min_lines: usize,
}

impl Default for LogCompressorConfig {
    fn default() -> Self {
        Self {
            max_lines: 200,
            context_lines: 2,
            min_lines: 50,
        }
    }
}

/// Content-aware compressor for build/test logs.
#[derive(Debug, Clone, Default)]
pub struct LogCompressor {
    config: LogCompressorConfig,
}

impl LogCompressor {
    pub fn new(config: LogCompressorConfig) -> Self {
        Self { config }
    }
}

impl ContentCompressor for LogCompressor {
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

        // Classify each line.
        let mut scored: Vec<(usize, f64, &str)> = lines
            .iter()
            .enumerate()
            .map(|(i, line)| (i, score_log_line(line, &query_words), *line))
            .collect();

        // Always keep first few and last few lines; mark them high score.
        let head = 5.min(lines.len());
        let tail = 5.min(lines.len());
        for s in scored.iter_mut().take(head) {
            s.1 += 2.0;
        }
        for s in scored.iter_mut().rev().take(tail) {
            s.1 += 2.0;
        }

        // Select top-scored lines, then restore original order.
        scored.sort_by(|a, b| crate::scoring::cmp_f64_desc(a.1, b.1));
        let max = self.config.max_lines;
        let mut keep: Vec<usize> = scored.iter().take(max).map(|(idx, _, _)| *idx).collect();

        // Add context around high-importance lines.
        let mut context = std::collections::BTreeSet::new();
        for &(idx, score, _) in &scored {
            if score >= 1.5 {
                let start = idx.saturating_sub(self.config.context_lines);
                let end = (idx + self.config.context_lines + 1).min(lines.len());
                for j in start..end {
                    context.insert(j);
                }
            }
        }
        keep.extend(context);
        keep.sort_unstable();
        keep.dedup();

        if keep.len() >= lines.len() {
            return content.to_string();
        }

        let mut out = String::new();
        for idx in keep {
            out.push_str(lines[idx]);
            out.push('\n');
        }
        out
    }
}

fn score_log_line(line: &str, query_words: &[String]) -> f64 {
    let lower = line.to_ascii_lowercase();
    let mut score = 0.0;

    if lower.contains("error")
        || lower.contains("fatal")
        || lower.contains("panic")
        || lower.contains("exception")
    {
        score += 3.0;
    } else if lower.contains("fail") || lower.contains("failed") {
        score += 2.5;
    } else if lower.contains("warn") {
        score += 1.5;
    } else if lower.contains("info") {
        score += 0.5;
    } else if lower.contains("debug") || lower.contains("trace") {
        score += 0.1;
    }

    // Timestamp heuristics.
    if lower.contains("t")
        && (line.contains("-") && line.contains(":"))
        && line.chars().any(|c| c.is_ascii_digit())
    {
        score += 0.2;
    }

    // Query overlap.
    for word in query_words {
        if lower.contains(word) {
            score += 0.5;
        }
    }

    score
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_log_passes_through() {
        let c = LogCompressor::default();
        let text = "INFO start\nERROR fail\n";
        assert_eq!(c.compress(text, None, 100), text);
    }

    #[test]
    fn long_log_compresses() {
        let c = LogCompressor::default();
        let mut text = String::new();
        for i in 0..300 {
            text.push_str(&format!("INFO step {}\n", i));
        }
        text.push_str("ERROR something broke\n");
        let out = c.compress(&text, Some("broke"), 1000);
        assert!(out.lines().count() < text.lines().count());
        assert!(out.contains("ERROR"));
    }
}
