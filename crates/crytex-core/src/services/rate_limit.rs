//! Rate limiting and retry decorator for inference backends.
//!
//! Wraps any [`InferenceManager`] with a per-backend token bucket and an
//! exponential-backoff retry policy. 429 (rate limited) and 5xx (transient)
//! responses are retried; after exhaustion the decorator surfaces a
//! [`InferenceError::RateLimited`] error.

use std::sync::Arc;

use async_trait::async_trait;
use crytex_inference::{
    BackendInfo, InferenceError, InferenceManager, InferenceRequest, InferenceResponse,
    LoRAAdapter, ModelInfo,
};
use tokio::sync::Mutex;
use tokio::time::{Duration, Instant};

/// Token-bucket rate limiter.
///
/// Tokens are replenished continuously at `rate_per_second` up to `capacity`.
/// `acquire` suspends the caller until the requested amount is available.
#[derive(Debug)]
pub struct TokenBucket {
    rate_per_second: f64,
    capacity: f64,
    tokens: Mutex<f64>,
    last_update: Mutex<Instant>,
}

impl TokenBucket {
    /// Create a bucket that allows `capacity` bursts and refills at
    /// `rate_per_second` tokens per second.
    pub fn new(rate_per_second: f64, capacity: f64) -> Self {
        Self {
            rate_per_second,
            capacity,
            tokens: Mutex::new(capacity),
            last_update: Mutex::new(Instant::now()),
        }
    }

    /// Acquire one token, waiting if the bucket is currently empty.
    pub async fn acquire(&self) {
        self.acquire_amount(1.0).await;
    }

    async fn acquire_amount(&self, amount: f64) {
        loop {
            let now = Instant::now();
            let mut tokens = self.tokens.lock().await;
            let mut last = self.last_update.lock().await;
            let elapsed = now.duration_since(*last).as_secs_f64();
            *tokens = (*tokens + elapsed * self.rate_per_second).min(self.capacity);
            *last = now;

            if *tokens >= amount {
                *tokens -= amount;
                return;
            }

            let deficit = amount - *tokens;
            let wait_ms = ((deficit / self.rate_per_second) * 1000.0).ceil() as u64;
            drop(tokens);
            drop(last);
            tokio::time::sleep(Duration::from_millis(wait_ms)).await;
        }
    }
}

/// Retry policy with exponential backoff.
#[derive(Debug, Clone, Copy)]
pub struct RetryPolicy {
    pub max_retries: u32,
    pub base_delay_ms: u64,
    pub max_delay_ms: u64,
    pub jitter: bool,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 3,
            base_delay_ms: 250,
            max_delay_ms: 8_000,
            jitter: false,
        }
    }
}

impl RetryPolicy {
    /// Delay before the `attempt`-th retry (1-indexed).
    pub fn delay_for_attempt(&self, attempt: u32) -> Duration {
        let exp = attempt.min(31) as u64;
        let base = self
            .base_delay_ms
            .saturating_mul(2_u64.saturating_pow(exp as u32));
        let delay_ms = base.min(self.max_delay_ms);
        Duration::from_millis(delay_ms)
    }
}

/// Decorator that applies rate limiting and retries to a backend.
pub struct RetryRateLimitBackend {
    inner: Arc<dyn InferenceManager>,
    limiter: Arc<TokenBucket>,
    retry_policy: RetryPolicy,
}

impl RetryRateLimitBackend {
    pub fn new(
        inner: Arc<dyn InferenceManager>,
        limiter: Arc<TokenBucket>,
        retry_policy: RetryPolicy,
    ) -> Self {
        Self {
            inner,
            limiter,
            retry_policy,
        }
    }

    /// Convenience constructor with a default token bucket (10 req/s, burst 10)
    /// and the default retry policy.
    pub fn default_for(inner: Arc<dyn InferenceManager>) -> Self {
        Self::new(
            inner,
            Arc::new(TokenBucket::new(10.0, 10.0)),
            RetryPolicy::default(),
        )
    }

    async fn call_with_retry<T, F, Fut>(&self, operation: F) -> Result<T, InferenceError>
    where
        F: Fn() -> Fut,
        Fut: std::future::Future<Output = Result<T, InferenceError>>,
    {
        self.limiter.acquire().await;

        let mut attempt = 0u32;
        loop {
            match operation().await {
                Ok(value) => return Ok(value),
                Err(InferenceError::RateLimited { retry_after_ms }) => {
                    if attempt >= self.retry_policy.max_retries {
                        return Err(InferenceError::RateLimited { retry_after_ms });
                    }
                    let delay = retry_after_ms
                        .map(Duration::from_millis)
                        .unwrap_or_else(|| self.retry_policy.delay_for_attempt(attempt + 1));
                    tokio::time::sleep(delay).await;
                    attempt += 1;
                }
                Err(InferenceError::Transient { status, body }) => {
                    if attempt >= self.retry_policy.max_retries {
                        return Err(InferenceError::Transient { status, body });
                    }
                    let delay = self.retry_policy.delay_for_attempt(attempt + 1);
                    tokio::time::sleep(delay).await;
                    attempt += 1;
                }
                Err(other) => return Err(other),
            }
        }
    }
}

#[async_trait]
impl InferenceManager for RetryRateLimitBackend {
    async fn generate(
        &self,
        request: InferenceRequest,
    ) -> Result<InferenceResponse, InferenceError> {
        self.call_with_retry(|| self.inner.generate(request.clone()))
            .await
    }

    async fn embed(&self, text: &str) -> Result<Vec<f32>, InferenceError> {
        let text = text.to_string();
        self.call_with_retry(|| self.inner.embed(&text)).await
    }

    async fn register_lora(&self, lora: LoRAAdapter) -> Result<(), InferenceError> {
        self.inner.register_lora(lora).await
    }

    async fn swap_lora(&self, lora_id: &str) -> Result<(), InferenceError> {
        self.inner.swap_lora(lora_id).await
    }

    fn available_backends(&self) -> Vec<BackendInfo> {
        self.inner.available_backends()
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, InferenceError> {
        self.call_with_retry(|| self.inner.list_models()).await
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    struct CountingBackend {
        successes_after: AtomicUsize,
        calls: AtomicUsize,
    }

    #[async_trait]
    impl InferenceManager for CountingBackend {
        async fn generate(
            &self,
            _request: InferenceRequest,
        ) -> Result<InferenceResponse, InferenceError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            if self.calls.load(Ordering::SeqCst) <= self.successes_after.load(Ordering::SeqCst) {
                Err(InferenceError::RateLimited {
                    retry_after_ms: None,
                })
            } else {
                Ok(InferenceResponse {
                    content: "ok".into(),
                    usage: crytex_inference::TokenUsage {
                        prompt_tokens: 0,
                        completion_tokens: 0,
                        total_tokens: 0,
                    },
                    finish_reason: "stop".into(),
                })
            }
        }

        async fn embed(&self, _text: &str) -> Result<Vec<f32>, InferenceError> {
            unimplemented!()
        }

        async fn register_lora(&self, _lora: LoRAAdapter) -> Result<(), InferenceError> {
            Ok(())
        }

        async fn swap_lora(&self, _lora_id: &str) -> Result<(), InferenceError> {
            Ok(())
        }

        fn available_backends(&self) -> Vec<BackendInfo> {
            vec![]
        }

        async fn list_models(&self) -> Result<Vec<ModelInfo>, InferenceError> {
            unimplemented!()
        }
    }

    fn dummy_request() -> InferenceRequest {
        InferenceRequest {
            backend_id: None,
            model: "mock".into(),
            messages: vec![],
            system_prompt: None,
            temperature: None,
            max_tokens: None,
            lora_adapter_id: None,
        }
    }

    #[tokio::test(start_paused = true)]
    async fn exponential_backoff_retries_429_then_succeeds() {
        let backend = Arc::new(CountingBackend {
            successes_after: AtomicUsize::new(2),
            calls: AtomicUsize::new(0),
        });
        let decorator = RetryRateLimitBackend::new(
            backend.clone(),
            Arc::new(TokenBucket::new(1000.0, 1000.0)),
            RetryPolicy {
                max_retries: 3,
                base_delay_ms: 10,
                max_delay_ms: 100,
                jitter: false,
            },
        );

        let result = decorator.generate(dummy_request()).await.unwrap();

        assert_eq!(result.content, "ok");
        assert_eq!(backend.calls.load(Ordering::SeqCst), 3);
    }

    #[tokio::test(start_paused = true)]
    async fn rate_limited_error_returned_when_retry_exhausted() {
        let backend = Arc::new(CountingBackend {
            successes_after: AtomicUsize::new(usize::MAX),
            calls: AtomicUsize::new(0),
        });
        let decorator = RetryRateLimitBackend::new(
            backend.clone(),
            Arc::new(TokenBucket::new(1000.0, 1000.0)),
            RetryPolicy {
                max_retries: 2,
                base_delay_ms: 10,
                max_delay_ms: 100,
                jitter: false,
            },
        );

        let err = decorator.generate(dummy_request()).await.unwrap_err();

        assert!(matches!(err, InferenceError::RateLimited { .. }));
        assert_eq!(backend.calls.load(Ordering::SeqCst), 3);
    }

    #[tokio::test(start_paused = true)]
    async fn rate_limiter_blocks_request_when_bucket_empty() {
        let backend = Arc::new(CountingBackend {
            successes_after: AtomicUsize::new(0),
            calls: AtomicUsize::new(0),
        });
        let decorator = RetryRateLimitBackend::new(
            backend.clone(),
            Arc::new(TokenBucket::new(0.5, 1.0)), // 1 token, refill every 2s
            RetryPolicy {
                max_retries: 0,
                base_delay_ms: 10,
                max_delay_ms: 100,
                jitter: false,
            },
        );

        let _first = decorator.generate(dummy_request()).await.unwrap();

        let second_handle = tokio::spawn(async move { decorator.generate(dummy_request()).await });

        tokio::time::advance(Duration::from_millis(100)).await;
        assert!(!second_handle.is_finished());

        tokio::time::advance(Duration::from_secs(3)).await;
        let result = second_handle.await.unwrap().unwrap();
        assert_eq!(result.content, "ok");
    }
}
