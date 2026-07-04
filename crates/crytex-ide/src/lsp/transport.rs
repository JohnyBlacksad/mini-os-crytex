//! Async JSON-RPC transport for LSP clients.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::VecDeque;
use std::sync::Arc;
use thiserror::Error;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{ChildStdin, ChildStdout};
use tokio::sync::Mutex;

/// Errors that can occur while communicating over an LSP transport.
#[derive(Debug, Error)]
pub enum LspError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("JSON-RPC error {code}: {message}")]
    JsonRpc {
        code: i32,
        message: String,
        data: Option<Value>,
    },
    #[error("unexpected message: {0}")]
    Unexpected(String),
    #[error("transport closed")]
    Closed,
}

/// A JSON-RPC request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<u64>,
    pub method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<Value>,
}

impl JsonRpcRequest {
    pub fn new(id: u64, method: &str, params: Option<Value>) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id: Some(id),
            method: method.to_string(),
            params,
        }
    }

    pub fn notification(method: &str, params: Option<Value>) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id: None,
            method: method.to_string(),
            params,
        }
    }
}

/// A JSON-RPC response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

/// A JSON-RPC error object.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcError {
    pub code: i32,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

/// Trait for async LSP transports.
#[async_trait]
pub trait LspTransport: Send + Sync {
    /// Send a raw JSON-RPC message (must already include headers for stdio).
    async fn send(&self, message: Value) -> Result<(), LspError>;

    /// Receive the next raw JSON-RPC message.
    async fn recv(&self) -> Result<Value, LspError>;
}

/// Transport over a child process's stdin/stdout.
pub struct StdioTransport {
    stdin: Arc<Mutex<ChildStdin>>,
    stdout: Arc<Mutex<BufReader<ChildStdout>>>,
}

impl StdioTransport {
    pub fn new(stdin: ChildStdin, stdout: ChildStdout) -> Self {
        Self {
            stdin: Arc::new(Mutex::new(stdin)),
            stdout: Arc::new(Mutex::new(BufReader::new(stdout))),
        }
    }
}

#[async_trait]
impl LspTransport for StdioTransport {
    async fn send(&self, message: Value) -> Result<(), LspError> {
        let body = serde_json::to_vec(&message)?;
        let header = format!("Content-Length: {}\r\n\r\n", body.len());
        let mut stdin = self.stdin.lock().await;
        stdin.write_all(header.as_bytes()).await?;
        stdin.write_all(&body).await?;
        stdin.flush().await?;
        Ok(())
    }

    async fn recv(&self) -> Result<Value, LspError> {
        let mut stdout = self.stdout.lock().await;
        let mut header = String::new();
        let mut content_length: Option<usize> = None;

        loop {
            header.clear();
            let bytes_read = stdout.read_line(&mut header).await?;
            if bytes_read == 0 {
                return Err(LspError::Closed);
            }
            let trimmed = header.trim();
            if trimmed.is_empty() {
                break;
            }
            if let Some(value) = trimmed.strip_prefix("Content-Length:") {
                content_length = value.trim().parse().ok();
            }
        }

        let length = content_length
            .ok_or_else(|| LspError::Unexpected("missing Content-Length header".into()))?;
        let mut buffer = vec![0u8; length];
        stdout.read_exact(&mut buffer).await?;
        let message = serde_json::from_slice(&buffer)?;
        Ok(message)
    }
}

/// In-memory transport for tests. The paired [`ChannelTransportHandle`] lets the
/// test inject responses and inspect requests sent by the client.
pub struct ChannelTransport {
    inbox: Arc<Mutex<VecDeque<Value>>>,
    outbox: Arc<Mutex<Vec<Value>>>,
}

/// Handle to a [`ChannelTransport`].
pub struct ChannelTransportHandle {
    inbox: Arc<Mutex<VecDeque<Value>>>,
    outbox: Arc<Mutex<Vec<Value>>>,
}

impl ChannelTransport {
    pub fn new() -> (Self, ChannelTransportHandle) {
        let inbox = Arc::new(Mutex::new(VecDeque::new()));
        let outbox = Arc::new(Mutex::new(Vec::new()));
        let transport = Self {
            inbox: inbox.clone(),
            outbox: outbox.clone(),
        };
        let handle = ChannelTransportHandle { inbox, outbox };
        (transport, handle)
    }
}

#[async_trait]
impl LspTransport for ChannelTransport {
    async fn send(&self, message: Value) -> Result<(), LspError> {
        self.outbox.lock().await.push(message);
        Ok(())
    }

    async fn recv(&self) -> Result<Value, LspError> {
        loop {
            if let Some(message) = self.inbox.lock().await.pop_front() {
                return Ok(message);
            }
            tokio::task::yield_now().await;
        }
    }
}

impl ChannelTransportHandle {
    /// Inject a raw JSON-RPC message to be received by the client.
    pub async fn inject(&self, message: Value) {
        self.inbox.lock().await.push_back(message);
    }

    /// Inject a JSON-RPC response with the given id and result.
    pub async fn inject_response(&self, id: u64, result: Value) {
        self.inject(
            serde_json::to_value(JsonRpcResponse {
                jsonrpc: "2.0".to_string(),
                id: Some(id),
                result: Some(result),
                error: None,
            })
            .unwrap(),
        )
        .await;
    }

    /// Drain all messages sent by the client.
    pub async fn drain_sent(&self) -> Vec<Value> {
        self.outbox.lock().await.drain(..).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn channel_transport_records_sent_messages() {
        let (transport, handle) = ChannelTransport::new();
        transport.send(json!({"hello": "world"})).await.unwrap();
        let sent = handle.drain_sent().await;
        assert_eq!(sent.len(), 1);
        assert_eq!(sent[0]["hello"], "world");
    }

    #[tokio::test]
    async fn channel_transport_round_trips_response() {
        let (transport, handle) = ChannelTransport::new();
        handle
            .inject(json!({"jsonrpc": "2.0", "id": 1, "result": 42}))
            .await;
        let msg = transport.recv().await.unwrap();
        assert_eq!(msg["id"], 1);
        assert_eq!(msg["result"], 42);
    }
}
