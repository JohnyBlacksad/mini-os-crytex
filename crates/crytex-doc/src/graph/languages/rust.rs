//! Rust code-graph extraction.

use std::collections::HashMap;

use petgraph::stable_graph::NodeIndex;
use tree_sitter::{Node, Tree};

use crate::graph::common::{child_by_kind, collect_nodes, node_span, node_text};
use crate::graph::languages::{
    ExtractedFile, LanguageExtractor, ReferenceInfo, add_file_node, count_lines, identifier_name,
    make_parser,
};
use crate::graph::{CodeGraph, EdgeKind, SymbolId, SymbolKind, SymbolNode};

pub struct RustExtractor;

impl RustExtractor {
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
        let id = SymbolNode::make_id("rust", kind, file_path, span, name);
        let signature = node_text(source, node)
            .lines()
            .next()
            .map(|s| s.trim().to_string());
        let idx = graph.add_symbol(SymbolNode {
            id: id.clone(),
            kind,
            name: name.into(),
            qualified_name: name.into(),
            language: "rust".into(),
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
    ) -> Option<SymbolId> {
        let name = identifier_name(node, source)?;
        let kind = if parent.is_some_and(|p| graph.graph[p].kind == SymbolKind::Struct) {
            SymbolKind::Method
        } else {
            SymbolKind::Function
        };
        let idx = self.add_symbol(graph, file_path, source, kind, node, name, parent, locals);

        // Collect calls inside the body.
        let mut calls = Vec::new();
        collect_nodes(
            *node,
            &["call_expression", "method_call_expression"],
            &mut calls,
        );
        for call in calls {
            if let Some((callee, confidence)) = callee_name(&call, source) {
                refs.push(ReferenceInfo {
                    caller_id: graph.graph[idx].id.clone(),
                    callee_name: callee,
                    kind: EdgeKind::Calls,
                    span: node_span(&call),
                    confidence,
                });
            }
        }
        Some(graph.graph[idx].id.clone())
    }

    fn extract_struct(
        &self,
        graph: &mut CodeGraph,
        file_path: &str,
        source: &str,
        node: &Node,
        parent: Option<NodeIndex>,
        locals: &mut HashMap<String, Vec<SymbolNode>>,
    ) -> Option<NodeIndex> {
        let name = identifier_name(node, source)?;
        let idx = self.add_symbol(
            graph,
            file_path,
            source,
            SymbolKind::Struct,
            node,
            name,
            parent,
            locals,
        );

        // Fields are contained by the struct.
        let mut fields = Vec::new();
        collect_nodes(*node, &["field_declaration"], &mut fields);
        for field in fields {
            if let Some(field_name) = identifier_name(&field, source) {
                self.add_symbol(
                    graph,
                    file_path,
                    source,
                    SymbolKind::Field,
                    &field,
                    field_name,
                    Some(idx),
                    locals,
                );
            }
        }
        Some(idx)
    }

    fn extract_enum(
        &self,
        graph: &mut CodeGraph,
        file_path: &str,
        source: &str,
        node: &Node,
        parent: Option<NodeIndex>,
        locals: &mut HashMap<String, Vec<SymbolNode>>,
    ) -> Option<NodeIndex> {
        let name = identifier_name(node, source)?;
        Some(self.add_symbol(
            graph,
            file_path,
            source,
            SymbolKind::Enum,
            node,
            name,
            parent,
            locals,
        ))
    }

    fn extract_trait(
        &self,
        graph: &mut CodeGraph,
        file_path: &str,
        source: &str,
        node: &Node,
        parent: Option<NodeIndex>,
        locals: &mut HashMap<String, Vec<SymbolNode>>,
    ) -> Option<NodeIndex> {
        let name = identifier_name(node, source)?;
        let idx = self.add_symbol(
            graph,
            file_path,
            source,
            SymbolKind::Trait,
            node,
            name,
            parent,
            locals,
        );

        // Trait methods are contained by the trait.
        let mut methods = Vec::new();
        collect_nodes(*node, &["function_item"], &mut methods);
        for method in methods {
            self.extract_function(
                graph,
                file_path,
                source,
                &method,
                Some(idx),
                locals,
                &mut Vec::new(),
            );
        }
        Some(idx)
    }

    fn extract_impl(
        &self,
        graph: &mut CodeGraph,
        file_path: &str,
        source: &str,
        node: &Node,
        locals: &mut HashMap<String, Vec<SymbolNode>>,
        refs: &mut Vec<ReferenceInfo>,
    ) -> Option<NodeIndex> {
        // impl Foo or impl Trait for Foo
        let mut type_identifiers = Vec::new();
        let mut has_for = false;
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            match child.kind() {
                "type_identifier" => type_identifiers.push(child),
                "for" => has_for = true,
                _ => {}
            }
        }

        let (type_name, trait_name) = if has_for {
            // First type identifier is the trait, second is the implementing type.
            if type_identifiers.len() >= 2 {
                (
                    node_text(source, &type_identifiers[1]),
                    Some(node_text(source, &type_identifiers[0])),
                )
            } else {
                return None;
            }
        } else {
            (node_text(source, type_identifiers.first()?), None)
        };

        let span = node_span(node);
        let id = SymbolNode::make_id("rust", SymbolKind::Struct, file_path, span, type_name);
        let idx = graph.add_symbol(SymbolNode {
            id: id.clone(),
            kind: SymbolKind::Struct,
            name: type_name.into(),
            qualified_name: type_name.into(),
            language: "rust".into(),
            file_path: file_path.into(),
            span,
            signature: None,
            docs: None,
            visibility: None,
        });
        locals
            .entry(type_name.into())
            .or_default()
            .push(graph.graph[idx].clone());

        // If a trait is present, add an Implements edge (heuristic confidence).
        if let Some(trait_name) = trait_name {
            refs.push(ReferenceInfo {
                caller_id: id.clone(),
                callee_name: trait_name.into(),
                kind: EdgeKind::Implements,
                span,
                confidence: 0.9,
            });
        }

        // Methods inside the impl are contained by the impl symbol.
        let mut methods = Vec::new();
        collect_nodes(*node, &["function_item"], &mut methods);
        for method in methods {
            self.extract_function(graph, file_path, source, &method, Some(idx), locals, refs);
        }
        Some(idx)
    }
}

impl LanguageExtractor for RustExtractor {
    fn language(&self) -> &'static str {
        "rust"
    }

    fn parse(&self, source: &str) -> Option<Tree> {
        let mut parser = make_parser(tree_sitter_rust::LANGUAGE.into());
        parser.parse(source, None)
    }

    fn extract(&self, file_path: &str, source: &str, graph: &mut CodeGraph) -> ExtractedFile {
        let file_idx = add_file_node(graph, "rust", file_path, count_lines(source));
        let mut locals: HashMap<String, Vec<SymbolNode>> = HashMap::new();
        let mut refs: Vec<ReferenceInfo> = Vec::new();

        let tree = self.parse(source).expect("rust parse");
        let root = tree.root_node();

        let mut cursor = root.walk();
        for child in root.children(&mut cursor) {
            match child.kind() {
                "function_item" | "function_signature_item" => {
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
                "struct_item" => {
                    self.extract_struct(
                        graph,
                        file_path,
                        source,
                        &child,
                        Some(file_idx),
                        &mut locals,
                    );
                }
                "enum_item" => {
                    self.extract_enum(
                        graph,
                        file_path,
                        source,
                        &child,
                        Some(file_idx),
                        &mut locals,
                    );
                }
                "trait_item" => {
                    self.extract_trait(
                        graph,
                        file_path,
                        source,
                        &child,
                        Some(file_idx),
                        &mut locals,
                    );
                }
                "impl_item" => {
                    self.extract_impl(graph, file_path, source, &child, &mut locals, &mut refs);
                }
                "mod_item" => {
                    if let Some(name) = identifier_name(&child, source) {
                        self.add_symbol(
                            graph,
                            file_path,
                            source,
                            SymbolKind::Module,
                            &child,
                            name,
                            Some(file_idx),
                            &mut locals,
                        );
                    }
                }
                "macro_definition" => {
                    if let Some(name) = identifier_name(&child, source) {
                        self.add_symbol(
                            graph,
                            file_path,
                            source,
                            SymbolKind::Macro,
                            &child,
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

/// Try to extract the callee name from a call or method call expression.
fn callee_name(node: &Node, source: &str) -> Option<(String, f32)> {
    match node.kind() {
        "call_expression" => {
            let func = child_by_kind(node, "identifier")
                .or_else(|| child_by_kind(node, "scoped_identifier"))
                .or_else(|| child_by_kind(node, "field_expression"))?;
            let text = node_text(source, &func);
            // scoped_identifier like `std::cmp::min` — take last segment
            let name = text.split("::").last()?.to_string();
            Some((name, 0.85))
        }
        "method_call_expression" => {
            let name = child_by_kind(node, "field_identifier")?;
            Some((node_text(source, &name).to_string(), 0.85))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn extract(source: &str) -> CodeGraph {
        let mut graph = CodeGraph::new();
        let extractor = RustExtractor;
        extractor.extract("src/lib.rs", source, &mut graph);
        graph
    }

    fn has_symbol(graph: &CodeGraph, name: &str, kind: SymbolKind) -> bool {
        graph.symbols().any(|s| s.name == name && s.kind == kind)
    }

    fn relation_count(graph: &CodeGraph, kind: EdgeKind) -> usize {
        graph
            .graph
            .edge_weights()
            .filter(|e| e.kind == kind)
            .count()
    }

    fn print_tree(node: &tree_sitter::Node, source: &str, depth: usize) {
        let indent = "  ".repeat(depth);
        println!(
            "{}{} [{}..{}] {:?}",
            indent,
            node.kind(),
            node.start_byte(),
            node.end_byte(),
            &source[node.start_byte().saturating_sub(0).min(source.len())
                ..node.end_byte().min(source.len())]
                .lines()
                .next()
                .unwrap_or("")
        );
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            print_tree(&child, source, depth + 1);
        }
    }

    #[test]
    fn rust_extractor_finds_function_calls() {
        let source = r#"
fn add(a: i32, b: i32) -> i32 { a + b }
fn main() { let x = add(1, 2); }
"#;
        let graph = extract(source);
        assert!(has_symbol(&graph, "add", SymbolKind::Function));
        assert!(has_symbol(&graph, "main", SymbolKind::Function));
        assert!(relation_count(&graph, EdgeKind::Calls) >= 1);
    }

    #[test]
    fn rust_extractor_finds_struct_definitions() {
        let source = r#"
pub struct Point { x: f64, y: f64 }
"#;
        let graph = extract(source);
        assert!(has_symbol(&graph, "Point", SymbolKind::Struct));
        assert!(has_symbol(&graph, "x", SymbolKind::Field));
        assert!(has_symbol(&graph, "y", SymbolKind::Field));
    }

    #[test]
    fn rust_extractor_finds_impl_relations() {
        let source = r#"
trait Greet { fn greet(&self); }
struct Person;
impl Greet for Person {
    fn greet(&self) { println!("hi"); }
}
"#;
        let mut parser = make_parser(tree_sitter_rust::LANGUAGE.into());
        let tree = parser.parse(source, None).unwrap();
        print_tree(&tree.root_node(), source, 0);
        let graph = extract(source);
        assert!(has_symbol(&graph, "Greet", SymbolKind::Trait));
        assert!(has_symbol(&graph, "Person", SymbolKind::Struct));
        assert!(has_symbol(&graph, "greet", SymbolKind::Method));
        assert!(relation_count(&graph, EdgeKind::Implements) >= 1);
    }
}
