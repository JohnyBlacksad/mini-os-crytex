//! Structured telemetry for Crytex.
//!
//! Provides JSON-formatted tracing with a `trace_id` correlation field that
//! propagates across service boundaries. The global subscriber writes to
//! stderr; tests can use a buffered layer via [`CrytexTelemetry::json_layer_with_writer`].

use tracing_subscriber::EnvFilter;
use tracing_subscriber::fmt::writer::MakeWriter;
use tracing_subscriber::layer::Layer;
use tracing_subscriber::prelude::*;

/// Global telemetry initializer.
pub struct CrytexTelemetry;

impl CrytexTelemetry {
    /// Initialize the global subscriber writing JSON logs to stderr.
    pub fn init() {
        let layer = Self::json_layer_with_writer(std::io::stderr);
        let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

        tracing_subscriber::registry()
            .with(layer.with_filter(filter))
            .init();
    }

    /// Build a JSON [`Layer`] that writes to the supplied writer.
    ///
    /// Useful in tests where a global subscriber cannot be re-initialized.
    /// Wrap the returned layer in [`tracing_subscriber::registry()`].
    pub fn json_layer_with_writer<W>(writer: W) -> impl Layer<tracing_subscriber::Registry>
    where
        W: for<'a> MakeWriter<'a> + Send + Sync + 'static,
    {
        tracing_subscriber::fmt::layer()
            .json()
            .flatten_event(true)
            .with_current_span(true)
            .with_span_list(false)
            .with_writer(writer)
    }
}

/// Correlation context carried through the call stack.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TraceContext {
    pub trace_id: String,
}

impl TraceContext {
    /// Create a new context with a fresh ULID trace id.
    pub fn new() -> Self {
        Self {
            trace_id: ulid::Ulid::new().to_string(),
        }
    }

    /// Returns a [`tracing::Span`] that carries the trace id.
    pub fn span(&self) -> tracing::Span {
        tracing::info_span!("trace", trace_id = %self.trace_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::sync::{Arc, Mutex};

    /// Test helper writer that accumulates log lines in a shared buffer.
    #[derive(Clone, Default)]
    struct TestWriter {
        buffer: Arc<Mutex<Vec<u8>>>,
    }

    impl TestWriter {
        fn take_string(&self) -> String {
            String::from_utf8(self.buffer.lock().unwrap().clone()).unwrap()
        }
    }

    impl<'a> MakeWriter<'a> for TestWriter {
        type Writer = TestWriter;

        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    impl Write for TestWriter {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.buffer.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn json_log_contains_trace_id() {
        // TDD PHASE: RED / GREEN / REFACTOR
        // Verifies that a TraceContext span injects trace_id into JSON output.
        let writer = TestWriter::default();
        let subscriber = tracing_subscriber::registry()
            .with(CrytexTelemetry::json_layer_with_writer(writer.clone()));
        let _guard = tracing::subscriber::set_default(subscriber);

        let ctx = TraceContext::new();
        {
            let span = ctx.span();
            let _enter = span.enter();
            tracing::info!(message = "hello from trace");
        }

        let logs = writer.take_string();
        assert!(
            logs.contains(&ctx.trace_id),
            "expected logs to contain trace_id {}, got: {}",
            ctx.trace_id,
            logs
        );
    }

    #[test]
    fn trace_context_generates_non_empty_id() {
        let ctx = TraceContext::new();
        assert!(!ctx.trace_id.is_empty());
    }
}
