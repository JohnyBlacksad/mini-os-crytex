//! Go code-graph extraction.

use std::collections::HashMap;

use petgraph::stable_graph::NodeIndex;
use tree_sitter::{Node, Tree};

use crate::graph::common::{child_by_kind, collect_nodes, node_span, node_text};
use crate::graph::languages::{
    ExtractedFile, LanguageExtractor, ReferenceInfo, add_file_node, count_lines, identifier_name,
    make_parser,
};
use crate::graph::{CodeGraph, EdgeKind, SymbolKind, SymbolNode};

pub struct GoExtractor;

impl GoExtractor {
    #[allow(clippy::too_many_arguments)]
    fn add_symbol(
        &self,
        graph: &mut CodeGraph,
        file_path: &str,
        source: &str,
        kind: SymbolKind,
        node: &Node,
        name: &str,
        parent: Option<NodeIndex>,
        locals: &mut HashMap<String, Vec<SymbolNode>>,
    ) -> NodeIndex {
        let span = node_span(node);
        let id = SymbolNode::make_id("go", kind, file_path, span, name);
        let signature = node_text(source, node)
            .lines()
            .next()
            .map(|s| s.trim().to_string());
        let idx = graph.add_symbol(SymbolNode {
            id: id.clone(),
            kind,
            name: name.into(),
            qualified_name: name.into(),
            language: "go".into(),
            file_path: file_path.into(),
            span,
            signature,
            docs: None,
            visibility: None,
        });
        if let Some(parent) = parent {
            graph.add_relation(parent, idx, EdgeKind::Contains, 1.0, Some(span));
        }
        locals
            .entry(name.into())
            .or_default()
            .push(graph.graph[idx].clone());
        idx
    }

    #[allow(clippy::too_many_arguments)]
    fn extract_function(
        &self,
        graph: &mut CodeGraph,
        file_path: &str,
        source: &str,
        node: &Node,
        parent: Option<NodeIndex>,
        locals: &mut HashMap<String, Vec<SymbolNode>>,
        refs: &mut Vec<ReferenceInfo>,
    ) -> Option<NodeIndex> {
        let name = identifier_name(node, source)?;
        let kind = if parent.is_some_and(|p| graph.graph[p].kind == SymbolKind::Class) {
            SymbolKind::Method
        } else {
            SymbolKind::Function
        };
        let idx = self.add_symbol(graph, file_path, source, kind, node, name, parent, locals);

        let mut calls = Vec::new();
        collect_nodes(*node, &["call_expression"], &mut calls);
        for call in calls {
            if let Some(func) = child_by_kind(&call, "identifier") {
                let callee = node_text(source, &func).to_string();
                refs.push(ReferenceInfo {
                    caller_id: graph.graph[idx].id.clone(),
                    callee_name: callee,
                    kind: EdgeKind::Calls,
                    span: node_span(&call),
                    confidence: 0.8,
                });
            }
        }
        Some(idx)
    }
}

impl LanguageExtractor for GoExtractor {
    fn language(&self) -> &'static str {
        "go"
    }

    fn parse(&self, source: &str) -> Option<Tree> {
        let mut parser = make_parser(tree_sitter_go::LANGUAGE.into());
        parser.parse(source, None)
    }

    fn extract(&self, file_path: &str, source: &str, graph: &mut CodeGraph) -> ExtractedFile {
        let file_idx = add_file_node(graph, "go", file_path, count_lines(source));
        let mut locals: HashMap<String, Vec<SymbolNode>> = HashMap::new();
        let mut refs: Vec<ReferenceInfo> = Vec::new();

        let tree = self.parse(source).expect("go parse");
        let root = tree.root_node();

        let mut cursor = root.walk();
        for child in root.children(&mut cursor) {
            match child.kind() {
                "function_declaration" => {
                    self.extract_function(
                        graph,
                        file_path,
                        source,
                        &child,
                        Some(file_idx),
                        &mut locals,
                        &mut refs,
                    );
                }
                "method_declaration" => {
                    self.extract_function(
                        graph,
                        file_path,
                        source,
                        &child,
                        Some(file_idx),
                        &mut locals,
                        &mut refs,
                    );
                }
                "type_declaration" => {
                    if let Some(spec) = child_by_kind(&child, "type_spec")
                        && let Some(name) = identifier_name(&spec, source)
                    {
                        self.add_symbol(
                            graph,
                            file_path,
                            source,
                            SymbolKind::Struct,
                            &spec,
                            name,
                            Some(file_idx),
                            &mut locals,
                        );
                    }
                }
                _ => {}
            }
        }

        crate::graph::languages::resolve_local_references(graph, &mut refs, &locals);
        ExtractedFile {
            file_idx,
            locals,
            references: refs,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn extract(source: &str) -> CodeGraph {
        let mut graph = CodeGraph::new();
        GoExtractor.extract("main.go", source, &mut graph);
        graph
    }

    fn has_symbol(graph: &CodeGraph, name: &str, kind: SymbolKind) -> bool {
        graph.symbols().any(|s| s.name == name && s.kind == kind)
    }

    #[test]
    fn go_extractor_finds_function_call() {
        let source = r#"
package main

func add(a, b int) int { return a + b }
func main() { add(1, 2) }
"#;
        let graph = extract(source);
        assert!(has_symbol(&graph, "add", SymbolKind::Function));
        assert!(has_symbol(&graph, "main", SymbolKind::Function));
        assert!(
            graph
                .graph
                .edge_weights()
                .any(|e| e.kind == EdgeKind::Calls)
        );
    }
}
