//! Graph-aware chunking.
//!
//! Links semantic code chunks to the code graph so that retrieval can be
//! expanded with callers, callees, and implementors.

use petgraph::stable_graph::NodeIndex;
use petgraph::visit::EdgeRef;

use crate::graph::{CodeGraph, SymbolId};
use crate::{Chunk, ChunkError, chunk_code};

/// Maximum number of related symbols to attach to a chunk.
const DEFAULT_RELATED_LIMIT: usize = 8;

/// Chunk `file_path` and attach graph metadata.
///
/// The returned chunks are the same as those produced by [`chunk_code`], but
/// each chunk gets:
/// - `symbol_id` of the symbol whose span overlaps the chunk,
/// - `related_symbols` taken from the symbol's direct callers, callees, and
///   implementors,
/// - a small header prepended to `text` describing scope and key relations.
pub fn chunk_code_with_graph(
    file_path: &str,
    source: &str,
    graph: &CodeGraph,
) -> Result<Vec<Chunk>, ChunkError> {
    let mut chunks = chunk_code(file_path, source)?;
    for chunk in &mut chunks {
        if let Some(symbol_idx) = find_symbol_for_chunk(graph, chunk) {
            let symbol = &graph.graph[symbol_idx];
            chunk.symbol_id = Some(symbol.id.clone());
            chunk.related_symbols =
                collect_related_symbols(graph, symbol_idx, DEFAULT_RELATED_LIMIT);
            chunk.text = prepend_scope_header(&chunk.text, symbol, &chunk.related_symbols, graph);
        }
    }
    Ok(chunks)
}

/// Find the symbol whose source span best overlaps the chunk.
fn find_symbol_for_chunk(graph: &CodeGraph, chunk: &Chunk) -> Option<NodeIndex> {
    graph
        .symbols()
        .filter(|s| s.file_path == chunk.source)
        .filter(|s| {
            // The chunk's line range overlaps the symbol's line range.
            chunk.start_line <= s.span.end_line && chunk.end_line >= s.span.start_line
        })
        .max_by_key(|s| {
            // Prefer the smallest overlapping symbol (most specific).
            let len = s.span.end_line.saturating_sub(s.span.start_line);
            std::cmp::Reverse(len)
        })
        .and_then(|s| graph.index.get(&s.id).copied())
}

/// Collect a bounded set of related symbols by graph distance.
fn collect_related_symbols(
    graph: &CodeGraph,
    symbol_idx: NodeIndex,
    limit: usize,
) -> Vec<SymbolId> {
    let mut related = Vec::new();
    let mut seen = std::collections::HashSet::new();
    seen.insert(symbol_idx);

    for edge in graph.graph.edges(symbol_idx) {
        let target = edge.target();
        if seen.insert(target) {
            related.push(graph.graph[target].id.clone());
            if related.len() >= limit {
                break;
            }
        }
    }

    // Also include direct callers.
    for edge in graph
        .graph
        .edges_directed(symbol_idx, petgraph::Direction::Incoming)
    {
        let source = edge.source();
        if seen.insert(source) {
            related.push(graph.graph[source].id.clone());
            if related.len() >= limit {
                break;
            }
        }
    }

    related
}

fn prepend_scope_header(
    text: &str,
    symbol: &crate::graph::SymbolNode,
    related: &[SymbolId],
    graph: &CodeGraph,
) -> String {
    let related_names: Vec<String> = related
        .iter()
        .filter_map(|id| graph.get(id))
        .map(|s| format!("{} {}", s.kind_as_str(), s.name))
        .collect();

    let mut header = format!(
        "// File: {}\n// Symbol: {} {}\n",
        symbol.file_path,
        symbol.kind_as_str(),
        symbol.name
    );
    if let Some(sig) = &symbol.signature {
        header.push_str(&format!("// Signature: {}\n", sig));
    }
    if !related_names.is_empty() {
        header.push_str(&format!("// Related: {}\n", related_names.join(", ")));
    }
    format!("{}\n{}", header, text)
}

impl crate::graph::SymbolNode {
    fn kind_as_str(&self) -> &'static str {
        match self.kind {
            crate::graph::SymbolKind::File => "file",
            crate::graph::SymbolKind::Module => "module",
            crate::graph::SymbolKind::Namespace => "namespace",
            crate::graph::SymbolKind::Function => "fn",
            crate::graph::SymbolKind::Method => "method",
            crate::graph::SymbolKind::Constructor => "constructor",
            crate::graph::SymbolKind::Struct => "struct",
            crate::graph::SymbolKind::Enum => "enum",
            crate::graph::SymbolKind::Trait => "trait",
            crate::graph::SymbolKind::Interface => "interface",
            crate::graph::SymbolKind::Class => "class",
            crate::graph::SymbolKind::Type => "type",
            crate::graph::SymbolKind::Field => "field",
            crate::graph::SymbolKind::Parameter => "param",
            crate::graph::SymbolKind::Variable => "var",
            crate::graph::SymbolKind::Constant => "const",
            crate::graph::SymbolKind::Macro => "macro",
            crate::graph::SymbolKind::Closure => "closure",
            crate::graph::SymbolKind::Unknown => "unknown",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::languages::LanguageExtractor;
    use crate::graph::languages::rust::RustExtractor;

    #[test]
    fn graph_chunk_includes_symbol_header() {
        let source = "fn add(a: i32, b: i32) -> i32 { a + b }\n";
        let mut graph = CodeGraph::new();
        RustExtractor.extract("src/lib.rs", source, &mut graph);

        let chunks = chunk_code_with_graph("src/lib.rs", source, &graph).unwrap();
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].symbol_id.is_some());
        assert!(chunks[0].text.contains("Symbol: fn add"));
    }

    #[test]
    fn graph_chunk_links_related_symbols() {
        let source = r#"
fn helper() {}
fn user() { helper(); }
"#;
        let mut graph = CodeGraph::new();
        RustExtractor.extract("src/lib.rs", source, &mut graph);

        let chunks = chunk_code_with_graph("src/lib.rs", source, &graph).unwrap();
        let helper_chunk = chunks
            .iter()
            .find(|c| c.text.contains("fn helper"))
            .expect("helper chunk");
        // The helper chunk should reference its caller(s).
        assert!(!helper_chunk.related_symbols.is_empty());
    }
}
