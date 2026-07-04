use std::collections::HashSet;

use tree_sitter::{Node, Parser};

use crate::compress::ContentCompressor;
use crate::tree_sitter_detector::LanguageId;

/// Configuration for [`CodeCompressor`].
#[derive(Debug, Clone)]
pub struct CodeCompressorConfig {
    /// Remove full-line comments.
    pub remove_comments: bool,
    /// Collapse runs of blank lines to a single blank line.
    pub collapse_blank_lines: bool,
    /// Minimum input lines before compression.
    pub min_lines: usize,
}

impl Default for CodeCompressorConfig {
    fn default() -> Self {
        Self {
            remove_comments: true,
            collapse_blank_lines: true,
            min_lines: 20,
        }
    }
}

const LANGUAGES: &[LanguageId] = &[
    LanguageId::Rust,
    LanguageId::Go,
    LanguageId::Python,
    LanguageId::JavaScript,
    LanguageId::TypeScript,
    LanguageId::Java,
    LanguageId::C,
    LanguageId::Cpp,
];

/// AST-aware compressor for source code.
#[derive(Debug, Clone, Default)]
pub struct CodeCompressor {
    config: CodeCompressorConfig,
}

impl CodeCompressor {
    pub fn new(config: CodeCompressorConfig) -> Self {
        Self { config }
    }
}

impl ContentCompressor for CodeCompressor {
    fn compress(&self, content: &str, _query: Option<&str>, _budget: usize) -> String {
        let lines: Vec<&str> = content.lines().collect();
        if lines.len() < self.config.min_lines {
            return content.to_string();
        }

        // Try all tree-sitter parsers and pick the one with the cleanest parse.
        let best = best_parse(content);
        let Some(tree) = best else {
            // Even without a clean parse we can still collapse blank lines.
            return collapse_blank_lines(&lines);
        };

        let mut comment_lines = HashSet::new();
        if self.config.remove_comments {
            collect_comment_lines(tree.root_node(), &mut comment_lines);
        }

        let mut output = Vec::new();
        let mut prev_blank = false;
        for (i, line) in lines.iter().enumerate() {
            let trimmed = line.trim();
            if self.config.remove_comments && comment_lines.contains(&i) {
                // Drop the line only if it becomes empty after removing the comment.
                let after = remove_comment_prefix(line);
                if after.trim().is_empty() {
                    continue;
                }
                output.push(after);
                prev_blank = false;
                continue;
            }
            if self.config.collapse_blank_lines {
                let blank = trimmed.is_empty();
                if blank && prev_blank {
                    continue;
                }
                prev_blank = blank;
            }
            output.push(*line);
        }

        if output.len() >= lines.len() {
            return collapse_blank_lines(&lines);
        }
        output.join("\n")
    }
}

fn best_parse(content: &str) -> Option<tree_sitter::Tree> {
    let mut best: Option<(tree_sitter::Tree, f64)> = None;
    for &id in LANGUAGES {
        let mut parser = Parser::new();
        let language = language_for(id);
        if parser.set_language(&language).is_err() {
            continue;
        }
        if let Some(tree) = parser.parse(content, None) {
            let root = tree.root_node();
            let total = root.descendant_count().max(1);
            let errors = count_errors(root);
            let error_rate = errors as f64 / total as f64;
            if error_rate <= 0.25
                && best
                    .as_ref()
                    .map(|(_, rate)| error_rate < *rate)
                    .unwrap_or(true)
            {
                best = Some((tree, error_rate));
            }
        }
    }
    best.map(|(tree, _)| tree)
}

fn count_errors(node: Node<'_>) -> usize {
    let mut count = if node.is_error() || node.is_missing() {
        1
    } else {
        0
    };
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        count += count_errors(child);
    }
    count
}

fn collapse_blank_lines(lines: &[&str]) -> String {
    let mut out = Vec::new();
    let mut prev_blank = false;
    for line in lines {
        let blank = line.trim().is_empty();
        if blank && prev_blank {
            continue;
        }
        prev_blank = blank;
        out.push(*line);
    }
    out.join("\n")
}

fn remove_comment_prefix(line: &str) -> &str {
    // For line comments, drop everything from // or # to end of line.
    if let Some(pos) = line.find("//") {
        return &line[..pos];
    }
    if let Some(pos) = line.find('#') {
        // Avoid hashing inside strings is acceptable for this fallback.
        return &line[..pos];
    }
    line
}

fn language_for(id: LanguageId) -> tree_sitter::Language {
    match id {
        LanguageId::Rust => tree_sitter_rust::LANGUAGE.into(),
        LanguageId::Python => tree_sitter_python::LANGUAGE.into(),
        LanguageId::JavaScript => tree_sitter_javascript::LANGUAGE.into(),
        LanguageId::TypeScript => tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
        LanguageId::Go => tree_sitter_go::LANGUAGE.into(),
        LanguageId::Java => tree_sitter_java::LANGUAGE.into(),
        LanguageId::C => tree_sitter_c::LANGUAGE.into(),
        LanguageId::Cpp => tree_sitter_cpp::LANGUAGE.into(),
    }
}

fn collect_comment_lines(node: Node<'_>, lines: &mut HashSet<usize>) {
    let kind = node.kind();
    if kind.contains("comment") {
        let start = node.start_position().row;
        let end = node.end_position().row;
        for i in start..=end {
            lines.insert(i);
        }
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        collect_comment_lines(child, lines);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_code_passes_through() {
        let c = CodeCompressor::default();
        let text = "fn main() { println!(); }";
        assert_eq!(c.compress(text, None, 100), text);
    }

    #[test]
    fn removes_comments_and_blanks() {
        let c = CodeCompressor::default();
        let mut text = String::from("fn main() {\n");
        for i in 0..30 {
            text.push_str(&format!(
                "    // comment {}\n    let x{} = {};\n\n",
                i, i, i
            ));
        }
        text.push_str("}\n");
        let out = c.compress(&text, None, 1000);
        assert!(out.lines().count() < text.lines().count());
        assert!(!out.contains("// comment"));
    }
}
