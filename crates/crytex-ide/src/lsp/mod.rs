//! LSP client support.

pub mod client;
pub mod server;
pub mod transport;

pub use client::LspClient;
pub use server::{LanguageServer, ServerCommand};
pub use transport::{
    ChannelTransport, ChannelTransportHandle, LspError, LspTransport, StdioTransport,
};
