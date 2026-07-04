//! Language-server process launcher.

use super::client::LspClient;
use super::transport::{LspError, StdioTransport};
use lsp_types::Uri;
use std::process::Stdio;
use url::Url;

/// Describes how to start a language server.
#[derive(Debug, Clone)]
pub struct ServerCommand {
    pub command: String,
    pub args: Vec<String>,
}

impl ServerCommand {
    pub fn new(command: &str, args: &[&str]) -> Self {
        Self {
            command: command.to_string(),
            args: args.iter().map(|s| s.to_string()).collect(),
        }
    }

    /// Built-in command registry for supported languages.
    pub fn for_language(language: &str) -> Option<Self> {
        match language.to_lowercase().as_str() {
            "rust" | "rust-analyzer" => Some(Self::new("rust-analyzer", &[])),
            "typescript" | "ts" | "javascript" | "js" => {
                Some(Self::new("typescript-language-server", &["--stdio"]))
            }
            "python" | "py" => Some(Self::new("pyright-langserver", &["--stdio"])),
            "go" | "golang" => Some(Self::new("gopls", &[])),
            _ => None,
        }
    }
}

/// Handle to a running language server process and its LSP client.
pub struct LanguageServer {
    command: ServerCommand,
}

impl LanguageServer {
    pub fn new(command: ServerCommand) -> Self {
        Self { command }
    }

    pub fn for_language(language: &str) -> Option<Self> {
        ServerCommand::for_language(language).map(Self::new)
    }

    /// Start the server process and return an [`LspClient`] connected to it.
    pub async fn start(&self, root_uri: Uri) -> Result<LspClient<StdioTransport>, LspError> {
        let root_path = Url::parse(root_uri.as_str())
            .ok()
            .and_then(|u| u.to_file_path().ok())
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| ".".into()));
        let mut child = tokio::process::Command::new(&self.command.command)
            .args(&self.command.args)
            .current_dir(root_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(LspError::Io)?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| LspError::Unexpected("failed to open server stdin".into()))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| LspError::Unexpected("failed to open server stdout".into()))?;

        let transport = StdioTransport::new(stdin, stdout);
        Ok(LspClient::new(transport))
    }
}
