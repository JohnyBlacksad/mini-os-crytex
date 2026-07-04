//! TypeScript / JavaScript code-graph extraction.

use std::collections::HashMap;

use petgraph::stable_graph::NodeIndex;
use tree_sitter::{Node, Tree};

use crate::graph::common::{child_by_kind, collect_nodes, node_span, node_text};
use crate::graph::languages::{
    ExtractedFile, LanguageExtractor, ReferenceInfo, add_file_node, count_lines, identifier_name,
    make_parser,
};
use crate::graph::{CodeGraph, EdgeKind, SymbolKind, SymbolNode};

pub struct TypeScriptExtractor;

impl TypeScriptExtractor {
    #[allow(clippy::too_many_arguments)]
    fn add_symbol(
        &self,
        graph: &mut CodeGraph,
        language: &str,
        file_path: &str,
        source: &str,
        kind: SymbolKind,
        node: &Node,
        name: &str,
        parent: Option<NodeIndex>,
        locals: &mut HashMap<String, Vec<SymbolNode>>,
    ) -> NodeIndex {
        let span = node_span(node);
        let id = SymbolNode::make_id(language, kind, file_path, span, name);
        let signature = node_text(source, node)
            .lines()
            .next()
            .map(|s| s.trim().to_string());
        let idx = graph.add_symbol(SymbolNode {
            id: id.clone(),
            kind,
            name: name.into(),
            qualified_name: name.into(),
            language: language.into(),
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
        language: &str,
        file_path: &str,
        source: &str,
        node: &Node,
        parent: Option<NodeIndex>,
        locals: &mut HashMap<String, Vec<SymbolNode>>,
        refs: &mut Vec<ReferenceInfo>,
    ) -> Option<NodeIndex> {
        let (kind, name) = match node.kind() {
            "function_declaration" | "function_signature" => (
                SymbolKind::Function,
                identifier_name(node, source)?.to_string(),
            ),
            "class_declaration" => (
                SymbolKind::Class,
                identifier_name(node, source)?.to_string(),
            ),
            "method_definition" => (
                SymbolKind::Method,
                identifier_name(node, source)?.to_string(),
            ),
            "arrow_function" => (SymbolKind::Closure, "closure".into()),
            _ => return None,
        };
        let idx = self.add_symbol(
            graph, language, file_path, source, kind, node, &name, parent, locals,
        );

        // Nested methods / functions and calls.
        let mut children = Vec::new();
        collect_nodes(
            *node,
            &[
                "function_declaration",
                "class_declaration",
                "method_definition",
                "arrow_function",
                "call_expression",
            ],
            &mut children,
        );
        for child in children {
            if child == *node {
                continue;
            }
            match child.kind() {
                "function_declaration"
                | "class_declaration"
                | "method_definition"
                | "arrow_function" => {
                    self.extract_definition(
                        graph,
                        language,
                        file_path,
                        source,
                        &child,
                        Some(idx),
                        locals,
                        refs,
                    );
                }
                "call_expression" => {
                    if let Some(callee) = self.callee_name(&child, source) {
                        refs.push(ReferenceInfo {
                            caller_id: graph.graph[idx].id.clone(),
                            callee_name: callee,
                            kind: EdgeKind::Calls,
                            span: node_span(&child),
                            confidence: 0.8,
                        });
                    }
                }
                _ => {}
            }
        }
        Some(idx)
    }

    fn callee_name(&self, node: &Node, source: &str) -> Option<String> {
        let func = child_by_kind(node, "identifier")
            .or_else(|| child_by_kind(node, "member_expression"))
            .or_else(|| child_by_kind(node, "call_expression"))?;
        let text = node_text(source, &func).to_string();
        // member_expression like `obj.method` — take last segment
        text.split('.')
            .next_back()
            .map(|s| s.to_string())
            .filter(|s| !s.is_empty())
    }
}

impl LanguageExtractor for TypeScriptExtractor {
    fn language(&self) -> &'static str {
        "typescript"
    }

    fn parse(&self, source: &str) -> Option<Tree> {
        // tree-sitter-typescript bundles both TS and TSX.
        let mut parser = make_parser(tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into());
        parser.parse(source, None)
    }

    fn extract(&self, file_path: &str, source: &str, graph: &mut CodeGraph) -> ExtractedFile {
        let language = if file_path.ends_with(".js") || file_path.ends_with(".jsx") {
            "javascript"
        } else {
            "typescript"
        };
        let file_idx = add_file_node(graph, language, file_path, count_lines(source));
        let mut locals: HashMap<String, Vec<SymbolNode>> = HashMap::new();
        let mut refs: Vec<ReferenceInfo> = Vec::new();

        let tree = self.parse(source).expect("typescript parse");
        let root = tree.root_node();

        let mut cursor = root.walk();
        for child in root.children(&mut cursor) {
            if matches!(
                child.kind(),
                "function_declaration"
                    | "class_declaration"
                    | "method_definition"
                    | "arrow_function"
            ) {
                self.extract_definition(
                    graph,
                    language,
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
        TypeScriptExtractor.extract("app.ts", source, &mut graph);
        graph
    }

    fn has_symbol(graph: &CodeGraph, name: &str, kind: SymbolKind) -> bool {
        graph.symbols().any(|s| s.name == name && s.kind == kind)
    }

    #[test]
    fn ts_extractor_finds_method_call() {
        let source = r#"
class Greeter {
    hello() { return "hi"; }
}
function run() {
    const g = new Greeter();
    return g.hello();
}
"#;
        let graph = extract(source);
        assert!(has_symbol(&graph, "Greeter", SymbolKind::Class));
        assert!(has_symbol(&graph, "hello", SymbolKind::Method));
        assert!(has_symbol(&graph, "run", SymbolKind::Function));
        assert!(
            graph
                .graph
                .edge_weights()
                .any(|e| e.kind == EdgeKind::Calls)
        );
    }
}
