//! Per-language AST extractors.

use std::collections::HashMap;

use petgraph::stable_graph::NodeIndex;
use tree_sitter::{Node, Parser, Tree};

use crate::graph::common::node_text;
use crate::graph::{CodeGraph, SourceSpan, SymbolId, SymbolKind, SymbolNode};

pub mod go;
pub mod python;
pub mod rust;
pub mod typescript;

/// Result of extracting symbols and relations from a single file.
#[derive(Debug, Clone, Default)]
pub struct ExtractedFile {
    /// The file symbol node index.
    pub file_idx: NodeIndex,
    /// Symbols defined in the file, keyed by name for intra-file resolution.
    pub locals: HashMap<String, Vec<SymbolNode>>,
    /// Unresolved references discovered during extraction.
    pub references: Vec<ReferenceInfo>,
}

/// Language-specific extraction logic.
pub trait LanguageExtractor: Send + Sync {
    /// Human-readable language name (must match `language_by_extension`).
    fn language(&self) -> &'static str;

    /// Parse `source` into a tree-sitter tree.
    fn parse(&self, source: &str) -> Option<Tree>;

    /// Extract all symbols and local relations from the file.
    ///
    /// Implementations should:
    /// - create a `File` node and add it to `graph`,
    /// - add all top-level and nested symbols,
    /// - add `Contains`/`Defines` edges where appropriate,
    /// - collect call/import references as `ReferenceInfo` for later linking.
    fn extract(&self, file_path: &str, source: &str, graph: &mut CodeGraph) -> ExtractedFile;
}

/// A reference discovered during extraction that needs to be linked later.
#[derive(Debug, Clone)]
pub struct ReferenceInfo {
    pub caller_id: SymbolId,
    pub callee_name: String,
    pub kind: crate::graph::EdgeKind,
    pub span: SourceSpan,
    pub confidence: f32,
}

/// Build a file symbol node and add it to the graph.
pub fn add_file_node(
    graph: &mut CodeGraph,
    language: &str,
    file_path: &str,
    lines: usize,
) -> NodeIndex {
    let span = SourceSpan {
        start_line: 1,
        start_col: 0,
        end_line: lines.max(1),
        end_col: 0,
    };
    let id = SymbolNode::make_id(language, SymbolKind::File, file_path, span, file_path);
    graph.add_symbol(SymbolNode {
        id,
        kind: SymbolKind::File,
        name: file_path.into(),
        qualified_name: file_path.into(),
        language: language.into(),
        file_path: file_path.into(),
        span,
        signature: None,
        docs: None,
        visibility: None,
    })
}

/// Utility: create a parser for a tree-sitter language.
pub fn make_parser(language: tree_sitter::Language) -> Parser {
    let mut parser = Parser::new();
    parser
        .set_language(&language)
        .expect("tree-sitter language");
    parser
}

/// Count the number of lines in a source string.
pub fn count_lines(source: &str) -> usize {
    source.lines().count()
}

/// Find a child identifier and return its text.
pub fn identifier_name<'a>(node: &Node, source: &'a str) -> Option<&'a str> {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if matches!(
            child.kind(),
            "identifier" | "type_identifier" | "property_identifier" | "field_identifier"
        ) {
            return Some(node_text(source, &child));
        }
    }
    None
}

/// Resolve references that point to symbols defined in the same file.
pub fn resolve_local_references(
    graph: &mut CodeGraph,
    refs: &mut Vec<ReferenceInfo>,
    locals: &HashMap<String, Vec<SymbolNode>>,
) {
    let mut unresolved = Vec::with_capacity(refs.len());
    for reference in refs.drain(..) {
        let caller_idx = match graph.index.get(&reference.caller_id) {
            Some(&idx) => idx,
            None => {
                unresolved.push(reference);
                continue;
            }
        };

        if let Some(candidates) = locals.get(&reference.callee_name) {
            let mut resolved = false;
            for candidate in candidates {
                if let Some(&target_idx) = graph.index.get(&candidate.id) {
                    graph.add_relation(
                        caller_idx,
                        target_idx,
                        reference.kind,
                        reference.confidence,
                        Some(reference.span),
                    );
                    resolved = true;
                }
            }
            if !resolved {
                unresolved.push(reference);
            }
        } else {
            unresolved.push(reference);
        }
    }
    *refs = unresolved;
}

/// Map file extensions to extractors.
pub fn extractor_for_extension(ext: &str) -> Option<Box<dyn LanguageExtractor>> {
    match ext {
        "rs" => Some(Box::new(rust::RustExtractor)),
        "py" => Some(Box::new(python::PythonExtractor)),
        "js" | "jsx" | "ts" | "tsx" => Some(Box::new(typescript::TypeScriptExtractor)),
        "go" => Some(Box::new(go::GoExtractor)),
        _ => None,
    }
}

/// Determine the language name for an extension, matching the rest of crytex-doc.
pub fn language_name(ext: &str) -> Option<&'static str> {
    match ext {
        "rs" => Some("rust"),
        "py" => Some("python"),
        "js" | "jsx" => Some("javascript"),
        "ts" | "tsx" => Some("typescript"),
        "go" => Some("go"),
        "java" => Some("java"),
        "c" | "h" => Some("c"),
        "cpp" | "cc" | "hpp" => Some("cpp"),
        _ => None,
    }
}
