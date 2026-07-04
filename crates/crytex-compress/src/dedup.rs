//! Generic deduplication helpers for text and structured content.
//!
//! Provides exact and near-duplicate detection via token-set Jaccard similarity
//! and 64-bit SimHash fingerprints.

use std::collections::HashSet;

use md5::{Digest, Md5};

/// Tokenize `text` into a set of normalized words.
pub fn word_set(text: &str) -> HashSet<String> {
    text.to_ascii_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| s.len() > 2)
        .map(String::from)
        .collect()
}

/// Jaccard similarity of two sets: `|A ∩ B| / |A ∪ B|`.
pub fn jaccard_similarity(a: &HashSet<String>, b: &HashSet<String>) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let intersection = a.intersection(b).count();
    let union = a.len() + b.len() - intersection;
    if union == 0 {
        return 0.0;
    }
    intersection as f64 / union as f64
}

/// Returns true if `candidate` is a near-duplicate of any item in `keepers`,
/// using SimHash with the given Hamming-distance threshold.
pub fn is_near_duplicate(candidate: &str, keepers: &[String], threshold: u32) -> bool {
    let fp = simhash(candidate);
    keepers
        .iter()
        .any(|seen| hamming_distance(fp, simhash(seen)) <= threshold)
}

/// Compute a 64-bit SimHash fingerprint for `text` using character 4-grams
/// hashed with MD5 and bit-voting.
pub fn simhash(text: &str) -> u64 {
    let lower = text.to_ascii_lowercase();
    let mut votes = [0i32; 64];
    let end = lower.len().saturating_sub(3).max(1);
    for i in 0..end {
        let gram = &lower[i..i + 4.min(lower.len() - i)];
        let hash = u64::from_be_bytes(Md5::digest(gram.as_bytes())[..8].try_into().unwrap());
        for (j, vote) in votes.iter_mut().enumerate() {
            if hash & (1 << j) != 0 {
                *vote += 1;
            } else {
                *vote -= 1;
            }
        }
    }
    let mut fingerprint = 0u64;
    for (j, &v) in votes.iter().enumerate() {
        if v > 0 {
            fingerprint |= 1 << j;
        }
    }
    fingerprint
}

/// Count differing bits between two 64-bit fingerprints.
pub fn hamming_distance(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}

/// Select indices of unique items from an ordered list, preserving order.
///
/// An item is considered a duplicate if its SimHash Hamming distance to any
/// previously selected item is at most `threshold`.
pub fn select_unique_indices(items: &[impl AsRef<str>], threshold: u32) -> Vec<usize> {
    let mut selected: Vec<usize> = Vec::new();
    let mut fingerprints: Vec<u64> = Vec::new();
    for (i, item) in items.iter().enumerate() {
        let fp = simhash(item.as_ref());
        let duplicate = fingerprints
            .iter()
            .any(|seen| hamming_distance(fp, *seen) <= threshold);
        if !duplicate {
            selected.push(i);
            fingerprints.push(fp);
        }
    }
    selected
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jaccard_identical_sets() {
        let a = word_set("rust compiler cargo");
        let b = word_set("rust compiler cargo");
        assert!((jaccard_similarity(&a, &b) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn jaccard_disjoint_sets() {
        let a = word_set("rust compiler");
        let b = word_set("python interpreter");
        assert_eq!(jaccard_similarity(&a, &b), 0.0);
    }

    #[test]
    fn simhash_detects_near_duplicates() {
        let a = "error: failed to build project on this machine with cargo";
        let b = "error: failed to build project on this machine with rustc";
        let c = "the weather is sunny today and tomorrow will be warm";
        let ab = hamming_distance(simhash(a), simhash(b));
        let ac = hamming_distance(simhash(a), simhash(c));
        assert!(
            ab < ac,
            "near-duplicate should be closer than unrelated: ab={}, ac={}",
            ab,
            ac
        );
    }

    #[test]
    fn select_unique_preserves_order() {
        let items = vec![
            "first unique message",
            "first unique message",
            "second unique message",
            "first unique message",
        ];
        let selected = select_unique_indices(&items, 3);
        assert_eq!(selected, vec![0, 2]);
    }
}
