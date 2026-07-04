//! Cache alignment detector.
//!
//! Detects volatile/dynamic content in the system prompt that would destabilize
//! the KV-cache prefix. This module never mutates prompts — it only emits
//! findings and observability metrics so callers can decide whether to move
//! dynamic values out of the system prompt.

use crate::message::Message;
use base64::Engine;

/// Classification label for a piece of volatile content.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VolatileLabel {
    Uuid,
    Iso8601,
    Jwt,
    HexHash,
}

impl VolatileLabel {
    pub fn as_str(&self) -> &'static str {
        match self {
            VolatileLabel::Uuid => "uuid",
            VolatileLabel::Iso8601 => "iso8601",
            VolatileLabel::Jwt => "jwt",
            VolatileLabel::HexHash => "hex_hash",
        }
    }
}

/// One detected piece of volatile content.
#[derive(Debug, Clone)]
pub struct VolatileFinding {
    pub label: VolatileLabel,
    /// Truncated sample; never includes the full secret verbatim.
    pub sample: String,
}

/// Result of analysing the system prompts of a message list.
#[derive(Debug, Clone)]
pub struct CacheAnalysis {
    pub findings: Vec<VolatileFinding>,
    /// Cache alignment score in `[0, 100]`. Higher means fewer volatile patterns.
    pub score: f64,
    /// Stable hash of the joined system prompts.
    pub prefix_hash: String,
    /// Whether the prefix hash changed since the previous analysis.
    pub prefix_changed: bool,
}

/// Detector-only cache aligner.
#[derive(Debug, Clone, Default)]
pub struct CacheAligner {
    previous_hash: Option<String>,
}

impl CacheAligner {
    pub fn new() -> Self {
        Self::default()
    }

    /// Detect volatile patterns in arbitrary text.
    pub fn detect(content: &str) -> Vec<VolatileFinding> {
        split_tokens(content)
            .into_iter()
            .filter_map(|token| {
                classify_token(&token).map(|label| {
                    let sample = if token.len() <= 16 {
                        token
                    } else {
                        format!("{}...{}", &token[..8], &token[token.len() - 4..])
                    };
                    VolatileFinding { label, sample }
                })
            })
            .collect()
    }

    /// Analyse system messages, update internal prefix hash, and return metrics.
    pub fn analyze(&mut self, messages: &[Message]) -> CacheAnalysis {
        let mut findings = Vec::new();
        for msg in messages {
            if msg.role == "system" {
                findings.extend(Self::detect(&msg.content));
            }
        }

        let system_text: String = messages
            .iter()
            .filter(|m| m.role == "system")
            .map(|m| m.content.as_str())
            .collect::<Vec<_>>()
            .join("\n---\n");

        let prefix_hash = crate::ccr::compute_key(&system_text);
        let prefix_changed = self
            .previous_hash
            .as_ref()
            .map(|prev| prev != &prefix_hash)
            .unwrap_or(false);
        self.previous_hash = Some(prefix_hash.clone());

        let mut score = 100.0_f64;
        for _ in &findings {
            score -= 10.0;
        }
        score = score.clamp(0.0, 100.0);

        CacheAnalysis {
            findings,
            score,
            prefix_hash,
            prefix_changed,
        }
    }

    /// Compute a cache alignment score without mutating internal state.
    pub fn score(messages: &[Message]) -> f64 {
        let count: usize = messages
            .iter()
            .filter(|m| m.role == "system")
            .map(|m| Self::detect(&m.content).len())
            .sum();
        (100.0 - count as f64 * 10.0).clamp(0.0, 100.0)
    }
}

fn split_tokens(content: &str) -> Vec<String> {
    content
        .split_whitespace()
        .map(|raw| raw.trim_matches(|c: char| ".;:!?\"'()[]<>{}".contains(c)))
        .filter(|s| !s.is_empty())
        .map(String::from)
        .collect()
}

fn classify_token(token: &str) -> Option<VolatileLabel> {
    if is_uuid(token) {
        return Some(VolatileLabel::Uuid);
    }
    if token.matches('.').count() == 2 && is_jwt_shape(token) {
        return Some(VolatileLabel::Jwt);
    }
    if is_iso8601(token) {
        return Some(VolatileLabel::Iso8601);
    }
    if is_hex_hash(token) {
        return Some(VolatileLabel::HexHash);
    }
    None
}

fn is_uuid(token: &str) -> bool {
    if token.len() != 36 || token.matches('-').count() != 4 {
        return false;
    }
    let parts: Vec<&str> = token.split('-').collect();
    if parts.len() != 5 {
        return false;
    }
    let lengths = [8, 4, 4, 4, 12];
    parts
        .iter()
        .zip(lengths.iter())
        .all(|(p, &len)| p.len() == len && p.chars().all(|c| c.is_ascii_hexdigit()))
}

fn is_jwt_shape(token: &str) -> bool {
    let segments: Vec<&str> = token.split('.').collect();
    if segments.len() != 3 {
        return false;
    }
    for seg in &segments {
        if seg.len() < 4 {
            return false;
        }
        let padded = pad_base64url(seg);
        if base64::prelude::BASE64_URL_SAFE.decode(&padded).is_err() {
            return false;
        }
    }
    true
}

fn pad_base64url(seg: &str) -> String {
    let pad = (4 - seg.len() % 4) % 4;
    let mut out = String::with_capacity(seg.len() + pad);
    out.push_str(seg);
    for _ in 0..pad {
        out.push('=');
    }
    out
}

fn is_iso8601(token: &str) -> bool {
    if token.len() < 8 {
        return false;
    }
    if !token.contains('T') && !token.contains('-') {
        return false;
    }
    parse_iso8601(token)
}

fn parse_iso8601(token: &str) -> bool {
    let candidate = token.strip_suffix('Z').unwrap_or(token);
    let (date_part, time_part) = match candidate.split_once('T') {
        Some(pair) => pair,
        None => (candidate, ""),
    };

    if !date_part.is_empty() && !parse_date(date_part) {
        return false;
    }
    if !time_part.is_empty() && !parse_time(time_part) {
        return false;
    }
    !date_part.is_empty() || !time_part.is_empty()
}

fn parse_date(date: &str) -> bool {
    let parts: Vec<&str> = date.split('-').collect();
    if parts.len() != 3 {
        return false;
    }
    let all_digits = parts.iter().all(|p| p.chars().all(|c| c.is_ascii_digit()));
    if !all_digits {
        return false;
    }
    let year = parts[0].parse::<u32>().unwrap_or(0);
    let month = parts[1].parse::<u32>().unwrap_or(0);
    let day = parts[2].parse::<u32>().unwrap_or(0);
    year > 0 && (1..=12).contains(&month) && (1..=31).contains(&day)
}

fn parse_time(time: &str) -> bool {
    // Strip timezone offset if present.
    let time = match time.split_once(['+', '-']) {
        Some((t, _)) => t,
        None => time,
    };
    let parts: Vec<&str> = time.split(':').collect();
    if parts.len() < 2 {
        return false;
    }
    if !parts
        .iter()
        .all(|p| p.chars().all(|c| c.is_ascii_digit() || c == '.'))
    {
        return false;
    }
    let hour = parts[0].parse::<u32>().unwrap_or(99);
    let minute = parts[1].parse::<u32>().unwrap_or(99);
    if hour > 23 || minute > 59 {
        return false;
    }
    if parts.len() >= 3 {
        let second = parts[2].parse::<f64>().unwrap_or(-1.0);
        if !(0.0..60.0).contains(&second) {
            return false;
        }
    }
    true
}

fn is_hex_hash(token: &str) -> bool {
    let len = token.len();
    if ![32, 40, 64].contains(&len) {
        return false;
    }
    token.chars().all(|c| c.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_uuid() {
        let text = "user id: 550e8400-e29b-41d4-a716-446655440000";
        let findings = CacheAligner::detect(text);
        assert!(findings.iter().any(|f| f.label == VolatileLabel::Uuid));
    }

    #[test]
    fn detects_iso8601() {
        let text = "timestamp: 2024-01-15T10:30:00Z";
        let findings = CacheAligner::detect(text);
        assert!(findings.iter().any(|f| f.label == VolatileLabel::Iso8601));
    }

    #[test]
    fn detects_jwt_shape() {
        // Header {"alg":"none"} payload {"sub":"x"} signature "sig" — all base64url.
        let text = "token: eyJhbGciOiJub25lIn0.eyJzdWIiOiJ4In0.c2ln";
        let findings = CacheAligner::detect(text);
        assert!(findings.iter().any(|f| f.label == VolatileLabel::Jwt));
    }

    #[test]
    fn detects_hex_hash() {
        let text = "hash: d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2";
        let findings = CacheAligner::detect(text);
        assert!(findings.iter().any(|f| f.label == VolatileLabel::HexHash));
    }

    #[test]
    fn analysis_tracks_prefix_changes() {
        let mut aligner = CacheAligner::new();
        let first = vec![Message::system("static prompt")];
        let a1 = aligner.analyze(&first);
        assert!(!a1.prefix_changed);

        let second = vec![Message::system("static prompt 2024-01-15T10:30:00Z")];
        let a2 = aligner.analyze(&second);
        assert!(a2.prefix_changed);
        assert!(
            a2.findings
                .iter()
                .any(|f| f.label == VolatileLabel::Iso8601)
        );
    }

    #[test]
    fn score_decreases_with_findings() {
        let messages = vec![Message::system(
            "uuid: 550e8400-e29b-41d4-a716-446655440000",
        )];
        let score = CacheAligner::score(&messages);
        assert_eq!(score, 90.0);
    }
}
