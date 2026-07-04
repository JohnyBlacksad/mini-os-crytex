//! High-level IDE service that orchestrates LSP clients and project state.

use crate::bridge::EditorBridge;
use crate::lsp::{LanguageServer, LspClient, LspError, LspTransport};
use crytex_core::persistence::ProjectSnapshotRepository;
use crytex_core::services::EventService;
use lsp_types::{ClientCapabilities, Location, ReferenceParams, TextDocumentPositionParams, Uri};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

/// High-level IDE service.
///
/// Owns per-project LSP clients and exposes editor-oriented operations.
pub struct IdeService {
    clients: Mutex<HashMap<String, Box<dyn LspClientHandle>>>,
}

/// Type-erased handle to an [`LspClient`] so different transports can live in one map.
#[async_trait::async_trait]
pub trait LspClientHandle: Send + Sync {
    async fn initialize(
        &self,
        root_uri: Uri,
        capabilities: ClientCapabilities,
    ) -> Result<(), LspError>;
    async fn definition(
        &self,
        params: TextDocumentPositionParams,
    ) -> Result<Option<lsp_types::GotoDefinitionResponse>, LspError>;
    async fn references(&self, params: ReferenceParams) -> Result<Option<Vec<Location>>, LspError>;
}

#[async_trait::async_trait]
impl<T: LspTransport + 'static> LspClientHandle for LspClient<T> {
    async fn initialize(
        &self,
        root_uri: Uri,
        capabilities: ClientCapabilities,
    ) -> Result<(), LspError> {
        let _ = LspClient::initialize(self, root_uri, capabilities).await?;
        Ok(())
    }

    async fn definition(
        &self,
        params: TextDocumentPositionParams,
    ) -> Result<Option<lsp_types::GotoDefinitionResponse>, LspError> {
        LspClient::goto_definition(self, params).await
    }

    async fn references(&self, params: ReferenceParams) -> Result<Option<Vec<Location>>, LspError> {
        LspClient::references(self, params).await
    }
}

impl IdeService {
    pub fn new() -> Self {
        Self {
            clients: Mutex::new(HashMap::new()),
        }
    }

    /// Start an LSP client for a project if one is not already running.
    pub async fn start_language_server(
        &self,
        project_id: &str,
        language: &str,
        root_uri: Uri,
    ) -> Result<(), LspError> {
        let mut clients = self.clients.lock().await;
        if clients.contains_key(project_id) {
            return Ok(());
        }

        let server = LanguageServer::for_language(language)
            .ok_or_else(|| LspError::Unexpected(format!("unsupported language: {language}")))?;
        let client = server.start(root_uri.clone()).await?;
        client
            .initialize(root_uri.clone(), ClientCapabilities::default())
            .await?;
        clients.insert(project_id.to_string(), Box::new(client));
        Ok(())
    }

    /// Request the definition of the symbol at the given position.
    pub async fn definition(
        &self,
        project_id: &str,
        params: TextDocumentPositionParams,
    ) -> Result<Option<lsp_types::GotoDefinitionResponse>, LspError> {
        let clients = self.clients.lock().await;
        let client = clients
            .get(project_id)
            .ok_or_else(|| LspError::Unexpected("no LSP client for project".into()))?;
        client.definition(params).await
    }

    /// Request references to the symbol at the given position.
    pub async fn references(
        &self,
        project_id: &str,
        params: ReferenceParams,
    ) -> Result<Option<Vec<Location>>, LspError> {
        let clients = self.clients.lock().await;
        let client = clients
            .get(project_id)
            .ok_or_else(|| LspError::Unexpected("no LSP client for project".into()))?;
        client.references(params).await
    }
}

impl Default for IdeService {
    fn default() -> Self {
        Self::new()
    }
}

/// Start the editor bridge and return both the bridge and the IDE service.
pub async fn start_ide_bridge(
    event_service: Arc<dyn EventService>,
    snapshots: Arc<dyn ProjectSnapshotRepository>,
) -> (EditorBridge, IdeService) {
    let bridge = EditorBridge::start(event_service, snapshots).await;
    let service = IdeService::new();
    (bridge, service)
}
