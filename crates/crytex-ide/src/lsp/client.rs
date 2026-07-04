//! LSP client built on top of a generic transport.

use super::transport::{JsonRpcRequest, JsonRpcResponse, LspError, LspTransport};
use lsp_types::{
    ClientCapabilities, GotoDefinitionResponse, InitializeParams, InitializeResult, Location,
    ReferenceParams, TextDocumentPositionParams, Uri, WorkspaceFolder,
};
use serde_json::Value;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// Async LSP client.
pub struct LspClient<T: LspTransport> {
    transport: Arc<T>,
    next_id: AtomicU64,
}

impl<T: LspTransport> LspClient<T> {
    pub fn new(transport: T) -> Self {
        Self {
            transport: Arc::new(transport),
            next_id: AtomicU64::new(1),
        }
    }

    fn next_id(&self) -> u64 {
        self.next_id.fetch_add(1, Ordering::SeqCst)
    }

    /// Initialize the language server.
    pub async fn initialize(
        &self,
        root_uri: Uri,
        capabilities: ClientCapabilities,
    ) -> Result<InitializeResult, LspError> {
        let params = InitializeParams {
            capabilities,
            workspace_folders: Some(vec![WorkspaceFolder {
                uri: root_uri,
                name: String::new(),
            }]),
            ..Default::default()
        };
        let result = self
            .request("initialize", Some(serde_json::to_value(params)?))
            .await?;
        let init: InitializeResult = serde_json::from_value(result)?;

        self.send_notification(
            "initialized",
            Some(serde_json::to_value(lsp_types::InitializedParams {})?),
        )
        .await?;

        Ok(init)
    }

    /// Ask the server to shut down and exit.
    pub async fn shutdown(&self) -> Result<(), LspError> {
        self.request("shutdown", None).await?;
        self.send_notification("exit", None).await?;
        Ok(())
    }

    /// Request `textDocument/definition`.
    pub async fn goto_definition(
        &self,
        params: TextDocumentPositionParams,
    ) -> Result<Option<GotoDefinitionResponse>, LspError> {
        let result = self
            .request(
                "textDocument/definition",
                Some(serde_json::to_value(params)?),
            )
            .await?;
        if result.is_null() {
            Ok(None)
        } else {
            Ok(Some(serde_json::from_value(result)?))
        }
    }

    /// Request `textDocument/references`.
    pub async fn references(
        &self,
        params: ReferenceParams,
    ) -> Result<Option<Vec<Location>>, LspError> {
        let result = self
            .request(
                "textDocument/references",
                Some(serde_json::to_value(params)?),
            )
            .await?;
        if result.is_null() {
            Ok(None)
        } else {
            Ok(Some(serde_json::from_value(result)?))
        }
    }

    async fn request(&self, method: &str, params: Option<Value>) -> Result<Value, LspError> {
        let id = self.next_id();
        let req = JsonRpcRequest::new(id, method, params);
        self.transport.send(serde_json::to_value(req)?).await?;

        loop {
            let msg = self.transport.recv().await?;
            if let Ok(resp) = serde_json::from_value::<JsonRpcResponse>(msg.clone())
                && resp.id == Some(id)
            {
                if let Some(err) = resp.error {
                    return Err(LspError::JsonRpc {
                        code: err.code,
                        message: err.message,
                        data: err.data,
                    });
                }
                return resp.result.ok_or(LspError::Unexpected(
                    "response contained neither result nor error".into(),
                ));
            }
            // Not the response we are waiting for; keep reading.
        }
    }

    async fn send_notification(&self, method: &str, params: Option<Value>) -> Result<(), LspError> {
        let req = JsonRpcRequest::notification(method, params);
        self.transport.send(serde_json::to_value(req)?).await
    }
}

#[cfg(test)]
mod tests {
    use super::super::transport::ChannelTransport;
    use super::*;
    use lsp_types::{Position, Range, TextDocumentIdentifier};
    use serde_json::json;
    use std::str::FromStr;

    async fn inject_response(
        handle: &super::super::transport::ChannelTransportHandle,
        id: u64,
        result: Value,
    ) {
        handle.inject_response(id, result).await;
    }

    #[tokio::test]
    async fn lsp_client_initializes_server() {
        let (transport, handle) = ChannelTransport::new();
        let client = LspClient::new(transport);

        let root = Uri::from_str("file:///project").unwrap();
        let caps = ClientCapabilities::default();

        inject_response(
            &handle,
            1,
            json!({
                "capabilities": { "definitionProvider": true },
                "serverInfo": { "name": "mock-lsp", "version": "1.0" }
            }),
        )
        .await;

        let result = client.initialize(root, caps).await.unwrap();
        assert_eq!(result.server_info.unwrap().name, "mock-lsp");

        let sent = handle.drain_sent().await;
        assert!(sent.iter().any(|m| m["method"] == "initialize"));
        assert!(sent.iter().any(|m| m["method"] == "initialized"));
    }

    #[tokio::test]
    async fn lsp_client_requests_definitions() {
        let (transport, handle) = ChannelTransport::new();
        let client = LspClient::new(transport);

        // initialize first
        inject_response(
            &handle,
            1,
            json!({ "capabilities": { "definitionProvider": true } }),
        )
        .await;
        client
            .initialize(
                Uri::from_str("file:///project").unwrap(),
                ClientCapabilities::default(),
            )
            .await
            .unwrap();

        let params = TextDocumentPositionParams {
            text_document: TextDocumentIdentifier {
                uri: Uri::from_str("file:///project/src/lib.rs").unwrap(),
            },
            position: Position::new(10, 5),
        };

        let loc = Location {
            uri: Uri::from_str("file:///project/src/foo.rs").unwrap(),
            range: Range::new(Position::new(3, 0), Position::new(3, 10)),
        };
        inject_response(&handle, 2, json!(loc)).await;

        let result = client.goto_definition(params).await.unwrap();
        let response = result.expect("definition response");
        match response {
            GotoDefinitionResponse::Scalar(location) => {
                assert_eq!(location.uri.path().as_str(), "/project/src/foo.rs");
            }
            _ => panic!("expected scalar location"),
        }
    }
}
