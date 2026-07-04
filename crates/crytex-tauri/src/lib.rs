//! Tauri IPC command scaffold for the Crytex desktop UI.
//!
//! This crate exposes plain async functions that mirror the commands the
//! Tauri frontend will invoke.  The functions intentionally depend on the
//! core service traits rather than the Tauri runtime so they can be unit
//! tested and reused outside of the Tauri process.

pub mod commands;

pub use commands::TauriCommandError;
