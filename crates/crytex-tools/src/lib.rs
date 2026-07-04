#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]

//! Agent tools for the Crytex kernel.
//!
//! Tools are capability-scoped operations that agents invoke through a JSON
//! schema interface. Each tool declares which permissions it needs; the runtime
//! enforces them through [`PermissionSet`] and [`PathSandbox`].

pub mod fs;
pub mod git;
pub mod parser;
pub mod policy;
pub mod process;
pub mod registry;
pub mod sandbox;
pub mod schema;
pub mod search;
pub mod service;

pub use fs::{FsList, FsRead, FsWrite};
pub use git::{GitCommit, GitDiff, GitStatus};
pub use parser::{ToolCall, parse_tool_calls};
pub use policy::{Capability, PermissionSet};
pub use process::RunCommand;
pub use registry::{ToolRegistry, TypedToolRegistry};
pub use sandbox::{PathSandbox, SandboxError, SandboxRoot};
pub use schema::{Tool, ToolError, ToolResult, ToolSchema};
pub use search::{SearchCode, SearchSemantic};
pub use service::{ScanningToolService, ToolServiceImpl};
