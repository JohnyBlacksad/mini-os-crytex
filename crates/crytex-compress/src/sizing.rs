//! Adaptive compression sizing via information saturation detection.
//!
//! Finds the "knee point" where adding more ranked items stops providing
//! meaningful new information, then validates the choice with SimHash
//! redundancy detection and zlib compression ratio.

use std::collections::HashSet;
use std::io::Write;

use flate2::Compression;
use flate2::write::ZlibEncoder;
use md5::{Digest, Md5};

/// Bias profile for the computed keep count.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SizingBias {
    /// Keep 50% more than the mathematical knee (`bias = 1.5`).
    Conservative,
    /// Trust the statistics (`bias = 1.0`).
    #[default]
    Moderate,
    /// Compress harder (`bias = 0.7`).
    Aggressive,
}

impl SizingBias {
    pub fn multiplier(&self) -> f64 {
        match self {
            SizingBias::Conservative => 1.5,
            SizingBias::Moderate => 1.0,
            SizingBias::Aggressive => 0.7,
        }
    }
}

/// Compute the optimal number of items to keep from an importance-ordered list.
///
/// * `items` – string representations of items, ordered from most to least
///   important.
/// * `bias` – multiplier on the knee point.
/// * `min_k` – never return fewer than this.
/// * `max_k` – optional upper bound.
pub fn optimal_k(
    items: &[impl AsRef<str>],
    bias: SizingBias,
    min_k: usize,
    max_k: Option<usize>,
) -> usize {
    let n = items.len();
    let effective_max = max_k.unwrap_or(n);

    // Tier 1: trivial cases.
    if n <= 8 {
        return n.min(effective_max);
    }

    let unique_count = count_unique_simhash(items);
    if unique_count <= 3 {
        let k = min_k.max(unique_count);
        return k.min(effective_max);
    }

    // Tier 2: Kneedle on unique bigram coverage.
    let curve = compute_unique_bigram_curve(items);
    let mut knee = find_knee(&curve);

    let diversity_ratio = unique_count as f64 / n as f64;

    if knee.is_none() {
        // No clear saturation: scale keep fraction with diversity.
        let keep_fraction = 0.3 + 0.7 * diversity_ratio;
        knee = Some(min_k.max((n as f64 * keep_fraction) as usize));
    } else if diversity_ratio > 0.7 {
        // High diversity: don't trust a shallow knee too much.
        let floor = min_k.max((n as f64 * (0.3 + 0.7 * diversity_ratio)) as usize);
        knee = Some(knee.unwrap().max(floor));
    }

    let mut k = min_k.max((knee.unwrap() as f64 * bias.multiplier()) as usize);
    k = k.min(effective_max);

    // Tier 3: validate with zlib compression ratio.
    k = validate_with_zlib(items, k, effective_max);
    k.min(effective_max).max(min_k)
}

fn find_knee(curve: &[usize]) -> Option<usize> {
    let n = curve.len();
    if n < 3 {
        return None;
    }
    let y_min = curve[0];
    let y_max = curve[n - 1];
    if y_max == y_min {
        return Some(1);
    }
    let y_range = (y_max - y_min) as f64;
    let x_range = (n - 1) as f64;

    let mut max_diff = -1.0_f64;
    let mut knee_idx = None;
    for (i, &y) in curve.iter().enumerate() {
        let x_norm = i as f64 / x_range;
        let y_norm = (y - y_min) as f64 / y_range;
        let diff = y_norm - x_norm;
        if diff > max_diff {
            max_diff = diff;
            knee_idx = Some(i);
        }
    }

    if max_diff < 0.05 {
        None
    } else {
        knee_idx.map(|i| i + 1)
    }
}

fn compute_unique_bigram_curve(items: &[impl AsRef<str>]) -> Vec<usize> {
    let mut seen: HashSet<(String, String)> = HashSet::new();
    items
        .iter()
        .map(|item| {
            let words: Vec<String> = item
                .as_ref()
                .to_ascii_lowercase()
                .split_whitespace()
                .map(String::from)
                .collect();
            if words.len() < 2 {
                seen.insert((words.first().cloned().unwrap_or_default(), String::new()));
            } else {
                for w in words.windows(2) {
                    seen.insert((w[0].clone(), w[1].clone()));
                }
            }
            seen.len()
        })
        .collect()
}

fn simhash(text: &str) -> u64 {
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

fn hamming_distance(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}

fn count_unique_simhash(items: &[impl AsRef<str>]) -> usize {
    let mut clusters: Vec<u64> = Vec::new();
    for item in items {
        let fp = simhash(item.as_ref());
        let matched = clusters.iter().any(|rep| hamming_distance(fp, *rep) <= 3);
        if !matched {
            clusters.push(fp);
        }
    }
    clusters.len()
}

fn validate_with_zlib(items: &[impl AsRef<str>], k: usize, max_k: usize) -> usize {
    if k >= items.len() || k >= max_k {
        return k;
    }
    let full_text = items
        .iter()
        .map(|s| s.as_ref())
        .collect::<Vec<_>>()
        .join("\n");
    let subset_text = items
        .iter()
        .take(k)
        .map(|s| s.as_ref())
        .collect::<Vec<_>>()
        .join("\n");
    if full_text.len() < 200 {
        return k;
    }

    let full_ratio = zlib_ratio(&full_text);
    let subset_ratio = zlib_ratio(&subset_text);
    let ratio_diff = (full_ratio - subset_ratio).abs();

    if ratio_diff > 0.15 {
        (k * 6 / 5).min(max_k)
    } else {
        k
    }
}

fn zlib_ratio(text: &str) -> f64 {
    let bytes = text.as_bytes();
    if bytes.is_empty() {
        return 1.0;
    }
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::fast());
    let _ = encoder.write_all(bytes);
    let compressed = encoder.finish().unwrap_or_default();
    compressed.len() as f64 / bytes.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_input_kept_intact() {
        let items: Vec<String> = (0..5).map(|i| format!("item {}", i)).collect();
        assert_eq!(optimal_k(&items, SizingBias::Moderate, 3, None), 5);
    }

    #[test]
    fn all_duplicates_yields_small_k() {
        let items: Vec<&str> = vec!["same text"; 100];
        let k = optimal_k(&items, SizingBias::Moderate, 3, None);
        assert!(k <= 3, "expected small k, got {}", k);
    }

    #[test]
    fn unique_items_kept_more_than_duplicates() {
        let unique: Vec<String> = (0..100)
            .map(|i| format!("unique content number {}", i))
            .collect();
        let duplicates: Vec<&str> = vec!["same text"; 100];
        let unique_k = optimal_k(&unique, SizingBias::Moderate, 3, None);
        let duplicate_k = optimal_k(&duplicates, SizingBias::Moderate, 3, None);
        assert!(
            unique_k > duplicate_k,
            "unique items should keep more than duplicates: {} vs {}",
            unique_k,
            duplicate_k
        );
    }

    #[test]
    fn bias_affects_result() {
        let items: Vec<String> = (0..50)
            .map(|i| format!("item {} with some shared words", i))
            .collect();
        let conservative = optimal_k(&items, SizingBias::Conservative, 3, None);
        let moderate = optimal_k(&items, SizingBias::Moderate, 3, None);
        let aggressive = optimal_k(&items, SizingBias::Aggressive, 3, None);
        assert!(conservative >= moderate && moderate >= aggressive);
    }

    #[test]
    fn max_k_is_respected() {
        let items: Vec<String> = (0..100).map(|i| format!("item {}", i)).collect();
        assert_eq!(optimal_k(&items, SizingBias::Moderate, 3, Some(10)), 10);
    }
}
