//! Shared tree-sitter helpers for graph extraction.

use tree_sitter::Node;

use crate::graph::SourceSpan;

/// Extract the source text covered by a node.
pub fn node_text<'a>(source: &'a str, node: &Node) -> &'a str {
    &source[node.start_byte()..node.end_byte()]
}

/// Convert a tree-sitter node to a `SourceSpan` (1-based lines, 0-based columns).
pub fn node_span(node: &Node) -> SourceSpan {
    let start = node.start_position();
    let end = node.end_position();
    SourceSpan {
        start_line: start.row + 1,
        start_col: start.column,
        end_line: end.row + 1,
        end_col: end.column,
    }
}

/// Find the first direct child with the given kind.
pub fn child_by_kind<'node>(node: &'node Node, kind: &str) -> Option<Node<'node>> {
    let mut cursor = node.walk();
    node.children(&mut cursor)
        .find(|&child| child.kind() == kind)
}

/// Collect all direct children with one of the given kinds.
#[allow(dead_code)]
pub fn children_by_kind<'node>(node: &'node Node, kinds: &[&str]) -> Vec<Node<'node>> {
    let mut cursor = node.walk();
    node.children(&mut cursor)
        .filter(|c| kinds.contains(&c.kind()))
        .collect()
}

/// Recursively collect nodes whose kind is in `target_kinds`.
pub fn collect_nodes<'node>(node: Node<'node>, target_kinds: &[&str], out: &mut Vec<Node<'node>>) {
    if target_kinds.contains(&node.kind()) {
        out.push(node);
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        collect_nodes(child, target_kinds, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_rust(source: &str) -> tree_sitter::Tree {
        let mut parser = tree_sitter::Parser::new();
        parser
            .set_language(&tree_sitter_rust::LANGUAGE.into())
            .unwrap();
        parser.parse(source, None).unwrap()
    }

    #[test]
    fn helpers_extract_identifiers() {
        let source = "fn add(a: i32, b: i32) -> i32 { a + b }";
        let tree = parse_rust(source);
        let root = tree.root_node();
        let func = child_by_kind(&root, "function_item").expect("function item");
        let name = child_by_kind(&func, "identifier").expect("identifier");
        assert_eq!(node_text(source, &name), "add");

        let params = children_by_kind(&func, &["parameters"]);
        assert_eq!(params.len(), 1);

        let mut identifiers = Vec::new();
        collect_nodes(func, &["identifier"], &mut identifiers);
        assert!(identifiers.len() >= 3); // add, a, b
    }
}
