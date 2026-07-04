use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;

use crate::compress::{CompressionError, Compressor};
use crate::message::Message;
use crate::token::TokenEstimator;

/// Configuration for [`Bm25Compressor`].
#[derive(Debug, Clone, Copy)]
pub struct Bm25Config {
    /// BM25 term frequency saturation parameter.
    pub k1: f64,
    /// BM25 length normalization parameter.
    pub b: f64,
}

impl Default for Bm25Config {
    fn default() -> Self {
        Self { k1: 1.5, b: 0.75 }
    }
}

/// Compressor that drops the least BM25-relevant messages until the conversation
/// fits the budget.
///
/// The latest user message is used as the query. System and the final message
/// are always protected.
#[derive(Clone)]
pub struct Bm25Compressor {
    estimator: Arc<dyn TokenEstimator>,
    config: Bm25Config,
}

impl Bm25Compressor {
    pub fn new(estimator: Arc<dyn TokenEstimator>) -> Self {
        Self {
            estimator,
            config: Bm25Config::default(),
        }
    }

    pub fn with_config(mut self, config: Bm25Config) -> Self {
        self.config = config;
        self
    }
}

impl std::fmt::Debug for Bm25Compressor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Bm25Compressor")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl Compressor for Bm25Compressor {
    async fn compress(
        &self,
        messages: &[Message],
        budget: usize,
    ) -> Result<Vec<Message>, CompressionError> {
        if messages.is_empty() {
            return Ok(vec![]);
        }

        let required = self
            .estimator
            .estimate_messages(messages)
            .unwrap_or(usize::MAX);
        if required <= budget {
            return Ok(messages.to_vec());
        }

        let system_index = messages.iter().position(|m| m.role == "system");
        let last_index = messages.len().saturating_sub(1);

        let query = messages
            .iter()
            .rev()
            .find(|m| m.role == "user")
            .map(|m| m.content.as_str())
            .unwrap_or("");
        let query_terms = tokenize(query);

        let docs: Vec<Vec<String>> = messages.iter().map(|m| tokenize(&m.content)).collect();
        let scores = bm25_scores(&query_terms, &docs, self.config.k1, self.config.b);

        let mut kept: Vec<usize> = (0..messages.len()).collect();

        while self.estimator.estimate_messages(
            &kept
                .iter()
                .map(|i| messages[*i].clone())
                .collect::<Vec<_>>(),
        )? > budget
        {
            let to_drop = kept
                .iter()
                .filter(|&&idx| Some(idx) != system_index && idx != last_index)
                .min_by(|&&a, &&b| crate::scoring::cmp_f64_asc(scores[a], scores[b]));

            match to_drop {
                Some(&idx) => {
                    kept.retain(|&i| i != idx);
                }
                None => break,
            }
        }

        kept.sort_unstable();
        Ok(kept.into_iter().map(|i| messages[i].clone()).collect())
    }
}

fn tokenize(text: &str) -> Vec<String> {
    text.to_ascii_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| s.len() > 2)
        .map(String::from)
        .collect()
}

fn bm25_scores(query: &[String], docs: &[Vec<String>], k1: f64, b: f64) -> Vec<f64> {
    let n = docs.len();
    if n == 0 || query.is_empty() {
        return vec![0.0; n];
    }

    let mut df: HashMap<String, usize> = HashMap::new();
    for doc in docs {
        let mut seen = std::collections::HashSet::new();
        for term in doc {
            if seen.insert(term.clone()) {
                *df.entry(term.clone()).or_insert(0) += 1;
            }
        }
    }

    let doc_lengths: Vec<f64> = docs.iter().map(|d| d.len() as f64).collect();
    let avgdl = doc_lengths.iter().sum::<f64>() / n as f64;
    let avgdl = if avgdl > 0.0 { avgdl } else { 1.0 };

    docs.iter()
        .enumerate()
        .map(|(i, doc)| {
            let mut term_freq: HashMap<String, usize> = HashMap::new();
            for term in doc {
                *term_freq.entry(term.clone()).or_insert(0) += 1;
            }
            let mut score = 0.0;
            for term in query {
                let f = *term_freq.get(term).unwrap_or(&0) as f64;
                if f == 0.0 {
                    continue;
                }
                let idf = ((n as f64 - df.get(term).copied().unwrap_or(0) as f64 + 0.5)
                    / (df.get(term).copied().unwrap_or(0) as f64 + 0.5)
                    + 1.0)
                    .ln();
                let denom = f + k1 * (1.0 - b + b * (doc_lengths[i] / avgdl));
                score += idf * (f * (k1 + 1.0)) / denom;
            }
            score
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::token::CharTokenEstimator;

    #[tokio::test]
    async fn keeps_relevant_messages() {
        let estimator = Arc::new(CharTokenEstimator);
        let compressor = Bm25Compressor::new(estimator);

        let messages = vec![
            Message::system("you are helpful"),
            Message::user("tell me about rust"),
            Message::assistant("rust is fast and memory safe"),
            Message::assistant("unrelated paragraph about gardening flowers soil"),
            Message::user("rust details"),
        ];

        let compressed = compressor.compress(&messages, 28).await.unwrap();
        assert_eq!(compressed.first().unwrap().role, "system");
        assert_eq!(compressed.last().unwrap().content, "rust details");
        assert!(
            compressed.iter().all(|m| !m.content.contains("gardening")),
            "least relevant message should be dropped"
        );
    }

    #[tokio::test]
    async fn returns_unchanged_when_under_budget() {
        let estimator = Arc::new(CharTokenEstimator);
        let compressor = Bm25Compressor::new(estimator);
        let messages = vec![Message::user("hello"), Message::assistant("hi there")];
        let compressed = compressor.compress(&messages, 1000).await.unwrap();
        assert_eq!(compressed.len(), 2);
    }
}
