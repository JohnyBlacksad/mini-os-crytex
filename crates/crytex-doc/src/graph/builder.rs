//! Build a [`CodeGraph`] by walking a project and extracting per-language symbols.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::graph::languages::{ExtractedFile, ReferenceInfo, extractor_for_extension};
use crate::graph::{CodeGraph, SymbolId, SymbolKind};
use crate::walk_project;

/// Configuration for building a code graph.
#[derive(Debug, Clone)]
pub struct GraphBuilderOptions {
    /// Maximum number of call-graph hops to resolve heuristically.
    pub max_call_depth: usize,
}

impl Default for GraphBuilderOptions {
    fn default() -> Self {
        Self { max_call_depth: 1 }
    }
}

/// Orchestrates project indexing into a code graph.
pub struct CodeGraphBuilder {
    options: GraphBuilderOptions,
}

impl Default for CodeGraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl CodeGraphBuilder {
    pub fn new() -> Self {
        Self {
            options: GraphBuilderOptions::default(),
        }
    }

    pub fn with_options(mut self, options: GraphBuilderOptions) -> Self {
        self.options = options;
        self
    }

    /// Index `project_root` and return the populated graph.
    pub fn index_project(&self, project_root: &Path) -> Result<CodeGraph, crate::ChunkError> {
        let files = walk_project(project_root)?;
        let mut graph = CodeGraph::new();
        let mut extracted: Vec<(PathBuf, ExtractedFile)> = Vec::new();

        for path in files {
            let relative = Path::new(&path)
                .strip_prefix(project_root)
                .unwrap_or(Path::new(&path))
                .to_string_lossy()
                .to_string()
                .replace('\\', "/");

            let source = match std::fs::read_to_string(&path) {
                Ok(s) => s,
                Err(_) => continue,
            };

            let ext = Path::new(&path)
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("");

            if let Some(extractor) = extractor_for_extension(ext) {
                let file = extractor.extract(&relative, &source, &mut graph);
                extracted.push((PathBuf::from(relative), file));
            }
        }

        // Build a global name -> symbol index for heuristic resolution.
        let mut global_index: HashMap<String, Vec<SymbolId>> = HashMap::new();
        for node in graph.symbols() {
            if matches!(
                node.kind,
                SymbolKind::Function
                    | SymbolKind::Method
                    | SymbolKind::Struct
                    | SymbolKind::Class
                    | SymbolKind::Trait
                    | SymbolKind::Enum
            ) {
                global_index
                    .entry(node.name.clone())
                    .or_default()
                    .push(node.id.clone());
            }
        }

        // Resolve references.
        for (_path, file) in &extracted {
            for reference in &file.references {
                resolve_reference(&mut graph, reference, &file.locals, &global_index);
            }
        }

        Ok(graph)
    }
}

fn resolve_reference(
    graph: &mut CodeGraph,
    reference: &ReferenceInfo,
    file_locals: &HashMap<String, Vec<crate::graph::SymbolNode>>,
    global_index: &HashMap<String, Vec<SymbolId>>,
) {
    let caller_idx = match graph.index.get(&reference.caller_id) {
        Some(&idx) => idx,
        None => return,
    };

    // Prefer a symbol defined in the same file with the same name.
    let candidates: Vec<SymbolId> = if let Some(locals) = file_locals.get(&reference.callee_name) {
        locals.iter().map(|n| n.id.clone()).collect()
    } else {
        global_index
            .get(&reference.callee_name)
            .cloned()
            .unwrap_or_default()
    };

    for target_id in candidates {
        if let Some(&target_idx) = graph.index.get(&target_id) {
            graph.add_relation(
                caller_idx,
                target_idx,
                reference.kind,
                reference.confidence,
                Some(reference.span),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::EdgeKind;
    use std::io::Write;

    fn make_project(files: &[(&str, &str)]) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        for (path, content) in files {
            let full = dir.path().join(path);
            if let Some(parent) = full.parent() {
                std::fs::create_dir_all(parent).unwrap();
            }
            let mut f = std::fs::File::create(full).unwrap();
            f.write_all(content.as_bytes()).unwrap();
        }
        dir
    }

    fn relation_count(graph: &CodeGraph, kind: EdgeKind) -> usize {
        graph
            .graph
            .edge_weights()
            .filter(|e| e.kind == kind)
            .count()
    }

    #[test]
    fn builder_indexes_sample_rust_project() {
        let dir = make_project(&[
            (".gitignore", "target/\n"),
            (
                "src/lib.rs",
                "fn add(a: i32, b: i32) -> i32 { a + b }\nfn main() { add(1, 2); }\n",
            ),
        ]);

        let builder = CodeGraphBuilder::new();
        let graph = builder.index_project(dir.path()).unwrap();

        assert!(graph.len() >= 3); // file + add + main
        assert!(relation_count(&graph, EdgeKind::Calls) >= 1);
    }
}
