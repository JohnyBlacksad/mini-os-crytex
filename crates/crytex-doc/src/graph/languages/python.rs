//! Python code-graph extraction.

use std::collections::HashMap;

use petgraph::stable_graph::NodeIndex;
use tree_sitter::{Node, Tree};

use crate::graph::common::{child_by_kind, collect_nodes, node_span, node_text};
use crate::graph::languages::{
    ExtractedFile, LanguageExtractor, ReferenceInfo, add_file_node, count_lines, identifier_name,
    make_parser,
};
use crate::graph::{CodeGraph, EdgeKind, SymbolId, SymbolKind, SymbolNode};

pub struct PythonExtractor;

impl PythonExtractor {
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
        let id = SymbolNode::make_id("python", kind, file_path, span, name);
        let signature = node_text(source, node)
            .lines()
            .next()
            .map(|s| s.trim().to_string());
        let idx = graph.add_symbol(SymbolNode {
            id: id.clone(),
            kind,
            name: name.into(),
            qualified_name: name.into(),
            language: "python".into(),
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
    fn extract_definition(
        &self,
        graph: &mut CodeGraph,
        file_path: &str,
        source: &str,
        node: &Node,
        parent: Option<NodeIndex>,
        locals: &mut HashMap<String, Vec<SymbolNode>>,
        refs: &mut Vec<ReferenceInfo>,
    ) -> Option<SymbolId> {
        let name = identifier_name(node, source)?;
        let kind = match node.kind() {
            "function_definition" => SymbolKind::Function,
            "class_definition" => SymbolKind::Class,
            _ => SymbolKind::Unknown,
        };
        let idx = self.add_symbol(graph, file_path, source, kind, node, name, parent, locals);

        // For classes, capture inheritance.
        if kind == SymbolKind::Class
            && let Some(bases) = child_by_kind(node, "argument_list")
        {
            let mut cursor = bases.walk();
            for base in bases.children(&mut cursor) {
                let base_name = node_text(source, &base).to_string();
                if !base_name.is_empty() {
                    refs.push(ReferenceInfo {
                        caller_id: graph.graph[idx].id.clone(),
                        callee_name: base_name,
                        kind: EdgeKind::Inherits,
                        span: node_span(&base),
                        confidence: 0.85,
                    });
                }
            }
        }

        // Nested definitions and calls.
        let mut children = Vec::new();
        collect_nodes(
            *node,
            &["function_definition", "class_definition", "call"],
            &mut children,
        );
        for child in children {
            if child == *node {
                continue;
            }
            match child.kind() {
                "function_definition" | "class_definition" => {
                    self.extract_definition(
                        graph,
                        file_path,
                        source,
                        &child,
                        Some(idx),
                        locals,
                        refs,
                    );
                }
                "call" => {
                    if let Some(func) = child_by_kind(&child, "identifier") {
                        let callee = node_text(source, &func).to_string();
                        refs.push(ReferenceInfo {
                            caller_id: graph.graph[idx].id.clone(),
                            callee_name: callee,
                            kind: EdgeKind::Calls,
                            span: node_span(&child),
                            confidence: 0.85,
                        });
                    }
                }
                _ => {}
            }
        }
        Some(graph.graph[idx].id.clone())
    }
}

impl LanguageExtractor for PythonExtractor {
    fn language(&self) -> &'static str {
        "python"
    }

    fn parse(&self, source: &str) -> Option<Tree> {
        let mut parser = make_parser(tree_sitter_python::LANGUAGE.into());
        parser.parse(source, None)
    }

    fn extract(&self, file_path: &str, source: &str, graph: &mut CodeGraph) -> ExtractedFile {
        let file_idx = add_file_node(graph, "python", file_path, count_lines(source));
        let mut locals: HashMap<String, Vec<SymbolNode>> = HashMap::new();
        let mut refs: Vec<ReferenceInfo> = Vec::new();

        let tree = self.parse(source).expect("python parse");
        let root = tree.root_node();

        let mut cursor = root.walk();
        for child in root.children(&mut cursor) {
            if child.kind() == "function_definition" || child.kind() == "class_definition" {
                self.extract_definition(
                    graph,
                    file_path,
                    source,
                    &child,
                    Some(file_idx),
                    &mut locals,
                    &mut refs,
                );
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
        PythonExtractor.extract("main.py", source, &mut graph);
        graph
    }

    fn has_symbol(graph: &CodeGraph, name: &str, kind: SymbolKind) -> bool {
        graph.symbols().any(|s| s.name == name && s.kind == kind)
    }

    #[test]
    fn python_extractor_finds_class_inheritance() {
        let source = r#"
class Animal:
    def speak(self): pass

class Dog(Animal):
    def bark(self): return self.speak()
"#;
        let graph = extract(source);
        assert!(has_symbol(&graph, "Animal", SymbolKind::Class));
        assert!(has_symbol(&graph, "Dog", SymbolKind::Class));
        assert!(
            graph
                .graph
                .edge_weights()
                .any(|e| e.kind == EdgeKind::Inherits)
        );
    }
}
