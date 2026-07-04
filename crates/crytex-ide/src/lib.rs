//! IDE / LSP bridge for Crytex.
//!
//! Provides an LSP client, an editor event bridge, and a protocol for
//! sending inline suggestions and diffs back to editor plugins.

pub mod bridge;
pub mod ide_service;
pub mod lsp;
pub mod protocol;

pub use bridge::{CursorPosition, EditorBridge, IdeProjectState, OpenFile};
pub use ide_service::{IdeService, start_ide_bridge};
pub use protocol::{
    DiffHunk, InlineSuggestionRequest, InlineSuggestionResponse, Suggestion, SuggestionAction,
    deserialize_request, serialize_suggestions,
};
