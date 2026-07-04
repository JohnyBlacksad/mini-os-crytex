//! Property graph over a project's source code.
//!
//! The graph captures symbols (functions, structs, traits, classes, modules, …)
//! and relations between them (contains, calls, implements, inherits, imports).
//! It is intentionally syntactic: it is built from tree-sitter ASTs without
//! invoking a full type checker. Heuristic edges are marked with a confidence
//! below `1.0` so they can be upgraded later with LSP/SCIP data.

use std::collections::{HashMap, HashSet};

pub mod builder;
pub(crate) mod common;
pub mod languages;

use petgraph::stable_graph::{NodeIndex, StableDiGraph};
use petgraph::visit::IntoEdgeReferences;
use serde::{Deserialize, Serialize};

/// Stable identifier for a symbol.
///
/// Format: `lang:kind:file_path:start_line:end_line:name`
/// The span component keeps the id stable across edits that do not move the
/// symbol, while the name makes it human-readable.
pub type SymbolId = String;

/// Kinds of symbols that can appear in the graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SymbolKind {
    File,
    Module,
    Namespace,
    Function,
    Method,
    Constructor,
    Struct,
    Enum,
    Trait,
    Interface,
    Class,
    Type,
    Field,
    Parameter,
    Variable,
    Constant,
    Macro,
    Closure,
    Unknown,
}

/// Kinds of relations between symbols.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EdgeKind {
    /// Parent contains child (file → symbol, impl → method, class → method).
    Contains,
    /// Scope defines a symbol (module → function).
    Defines,
    /// One callable calls another.
    Calls,
    /// A type implements a trait/interface.
    Implements,
    /// A type/class inherits from another.
    Inherits,
    /// A file/scope imports another module or symbol.
    Imports,
    /// Any reference that is not a call or import.
    References,
}

/// A source location.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SourceSpan {
    pub start_line: usize,
    pub start_col: usize,
    pub end_line: usize,
    pub end_col: usize,
}

/// A node in the code graph.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SymbolNode {
    pub id: SymbolId,
    pub kind: SymbolKind,
    pub name: String,
    pub qualified_name: String,
    pub language: String,
    pub file_path: String,
    pub span: SourceSpan,
    pub signature: Option<String>,
    pub docs: Option<String>,
    pub visibility: Option<String>,
}

impl SymbolNode {
    /// Construct a stable symbol id from its components.
    pub fn make_id(
        language: &str,
        kind: SymbolKind,
        file_path: &str,
        span: SourceSpan,
        name: &str,
    ) -> SymbolId {
        format!(
            "{}:{:?}:{}:{}:{}:{}",
            language, kind, file_path, span.start_line, span.end_line, name
        )
    }
}

/// An edge in the code graph.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RelationEdge {
    pub kind: EdgeKind,
    /// 1.0 for edges derived directly from the AST; lower for heuristics.
    pub confidence: f32,
    pub span: Option<SourceSpan>,
}

/// In-memory code graph.
#[derive(Clone)]
pub struct CodeGraph {
    pub graph: StableDiGraph<SymbolNode, RelationEdge>,
    pub index: HashMap<SymbolId, NodeIndex>,
}

impl Default for CodeGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl CodeGraph {
    pub fn new() -> Self {
        Self {
            graph: StableDiGraph::new(),
            index: HashMap::new(),
        }
    }

    /// Add a symbol node, or return the existing node if the id is already present.
    pub fn add_symbol(&mut self, node: SymbolNode) -> NodeIndex {
        if let Some(&idx) = self.index.get(&node.id) {
            return idx;
        }
        let id = node.id.clone();
        let idx = self.graph.add_node(node);
        self.index.insert(id, idx);
        idx
    }

    /// Add a directed relation between two symbols.
    pub fn add_relation(
        &mut self,
        source: NodeIndex,
        target: NodeIndex,
        kind: EdgeKind,
        confidence: f32,
        span: Option<SourceSpan>,
    ) {
        self.graph.add_edge(
            source,
            target,
            RelationEdge {
                kind,
                confidence,
                span,
            },
        );
    }

    /// Look up a symbol by id.
    pub fn get(&self, id: &SymbolId) -> Option<&SymbolNode> {
        self.index.get(id).map(|&idx| &self.graph[idx])
    }

    /// Iterator over all symbol nodes.
    pub fn symbols(&self) -> impl Iterator<Item = &SymbolNode> {
        self.graph.node_weights()
    }

    /// Number of symbols in the graph.
    pub fn len(&self) -> usize {
        self.graph.node_count()
    }

    pub fn is_empty(&self) -> bool {
        self.graph.node_count() == 0
    }

    /// Render a compact Markdown summary of the codebase graph.
    ///
    /// The summary is intentionally concise so it fits into an LLM prompt:
    /// file/symbol/relation counts, top-level modules, the public API surface,
    /// and relation counts.
    pub fn summary(&self) -> String {
        let mut parts = Vec::new();
        parts.push("# Codebase Map".to_string());

        let file_set: HashSet<&str> = self.symbols().map(|s| s.file_path.as_str()).collect();
        parts.push(format!("- Files: {}", file_set.len()));
        parts.push(format!("- Symbols: {}", self.len()));
        parts.push(format!("- Relations: {}", self.graph.edge_count()));

        let mut counts: HashMap<SymbolKind, usize> = HashMap::new();
        for sym in self.symbols() {
            *counts.entry(sym.kind).or_insert(0) += 1;
        }
        let mut kinds: Vec<_> = counts.iter().collect();
        kinds.sort_by(|a, b| b.1.cmp(a.1));
        parts.push("\n## Symbols by kind".to_string());
        for (kind, count) in kinds {
            parts.push(format!("- {:?}: {}", kind, count));
        }

        let mut top_level: Vec<&SymbolNode> = self
            .symbols()
            .filter(|s| s.kind == SymbolKind::Module || s.kind == SymbolKind::File)
            .collect();
        top_level.sort_by_key(|s| &s.qualified_name);
        if !top_level.is_empty() {
            parts.push("\n## Top-level modules/files".to_string());
            for sym in top_level.iter().take(50) {
                parts.push(format!("- `{}` ({})", sym.qualified_name, sym.language));
            }
        }

        let interesting = [
            SymbolKind::Function,
            SymbolKind::Method,
            SymbolKind::Struct,
            SymbolKind::Enum,
            SymbolKind::Trait,
            SymbolKind::Interface,
            SymbolKind::Class,
            SymbolKind::Type,
        ];
        let mut public_api: Vec<&SymbolNode> = self
            .symbols()
            .filter(|s| {
                interesting.contains(&s.kind)
                    && s.visibility
                        .as_deref()
                        .map(|v| v == "public" || v == "pub")
                        .unwrap_or(false)
            })
            .collect();
        public_api.sort_by(|a, b| {
            a.file_path
                .cmp(&b.file_path)
                .then(a.span.start_line.cmp(&b.span.start_line))
        });
        if !public_api.is_empty() {
            parts.push("\n## Public API surface".to_string());
            for sym in public_api.iter().take(100) {
                parts.push(format!(
                    "- {:?} `{}` ({}:{})",
                    sym.kind, sym.qualified_name, sym.file_path, sym.span.start_line
                ));
            }
        }

        let mut edge_counts: HashMap<EdgeKind, usize> = HashMap::new();
        for e in self.graph.edge_references() {
            *edge_counts.entry(e.weight().kind).or_insert(0) += 1;
        }
        let mut edge_kinds: Vec<_> = edge_counts.iter().collect();
        edge_kinds.sort_by(|a, b| b.1.cmp(a.1));
        if !edge_kinds.is_empty() {
            parts.push("\n## Relation counts".to_string());
            for (kind, count) in edge_kinds {
                parts.push(format!("- {:?}: {}", kind, count));
            }
        }

        parts.join("\n")
    }
}

impl std::fmt::Debug for CodeGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CodeGraph")
            .field("symbols", &self.graph.node_count())
            .field("relations", &self.graph.edge_count())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_span() -> SourceSpan {
        SourceSpan {
            start_line: 10,
            start_col: 0,
            end_line: 12,
            end_col: 1,
        }
    }

    fn sample_node(name: &str) -> SymbolNode {
        let span = sample_span();
        SymbolNode {
            id: SymbolNode::make_id("rust", SymbolKind::Function, "src/lib.rs", span, name),
            kind: SymbolKind::Function,
            name: name.into(),
            qualified_name: format!("crate::{}", name),
            language: "rust".into(),
            file_path: "src/lib.rs".into(),
            span,
            signature: None,
            docs: None,
            visibility: None,
        }
    }

    #[test]
    fn symbol_id_is_stable_for_same_span() {
        let id1 = SymbolNode::make_id(
            "rust",
            SymbolKind::Function,
            "src/lib.rs",
            sample_span(),
            "foo",
        );
        let id2 = SymbolNode::make_id(
            "rust",
            SymbolKind::Function,
            "src/lib.rs",
            sample_span(),
            "foo",
        );
        assert_eq!(id1, id2);
    }

    #[test]
    fn add_symbol_avoids_duplicates() {
        let mut graph = CodeGraph::new();
        let idx1 = graph.add_symbol(sample_node("foo"));
        let idx2 = graph.add_symbol(sample_node("foo"));
        assert_eq!(idx1, idx2);
        assert_eq!(graph.len(), 1);
    }

    fn node_with_kind(name: &str, kind: SymbolKind, visibility: Option<&str>) -> SymbolNode {
        let span = sample_span();
        SymbolNode {
            id: SymbolNode::make_id("rust", kind, "src/lib.rs", span, name),
            kind,
            name: name.into(),
            qualified_name: format!("crate::{}", name),
            language: "rust".into(),
            file_path: "src/lib.rs".into(),
            span,
            signature: None,
            docs: None,
            visibility: visibility.map(|s| s.to_string()),
        }
    }

    #[test]
    fn graph_summary_includes_top_level_symbols() {
        let mut graph = CodeGraph::new();
        graph.add_symbol(node_with_kind("src/lib.rs", SymbolKind::File, None));
        graph.add_symbol(node_with_kind("my_mod", SymbolKind::Module, None));
        graph.add_symbol(node_with_kind("MyStruct", SymbolKind::Struct, Some("pub")));
        graph.add_symbol(node_with_kind(
            "do_thing",
            SymbolKind::Function,
            Some("pub"),
        ));
        graph.add_symbol(node_with_kind(
            "helper",
            SymbolKind::Function,
            Some("private"),
        ));

        let summary = graph.summary();
        assert!(summary.contains("# Codebase Map"));
        assert!(summary.contains("Files: 1"));
        assert!(summary.contains("Symbols: 5"));
        assert!(summary.contains("- Struct: 1"));
        assert!(summary.contains("- Function: 2"));
        assert!(summary.contains("`crate::my_mod`"));
        assert!(summary.contains("`crate::MyStruct`"));
        assert!(summary.contains("`crate::do_thing`"));
        assert!(!summary.contains("helper"));
    }
}
