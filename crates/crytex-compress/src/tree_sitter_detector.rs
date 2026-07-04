use tree_sitter::{Language, Parser};

/// Supported programming languages for detection and AST-based compression.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LanguageId {
    Rust,
    Python,
    JavaScript,
    TypeScript,
    Go,
    Java,
    C,
    Cpp,
}

impl LanguageId {
    pub fn as_str(&self) -> &'static str {
        match self {
            LanguageId::Rust => "rust",
            LanguageId::Python => "python",
            LanguageId::JavaScript => "javascript",
            LanguageId::TypeScript => "typescript",
            LanguageId::Go => "go",
            LanguageId::Java => "java",
            LanguageId::C => "c",
            LanguageId::Cpp => "cpp",
        }
    }
}

fn language(id: LanguageId) -> Language {
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

/// Try to detect which programming language `text` is written in.
///
/// Combines tree-sitter parse quality with language-specific keyword hints,
/// so small snippets are recognized correctly instead of being eaten by the
/// most permissive parser.
pub fn detect_language(text: &str) -> Option<LanguageId> {
    if text.len() < 30 {
        return None;
    }

    if !looks_like_code(text) {
        return None;
    }

    let mut candidates: Vec<(LanguageId, f64)> = Vec::new();
    for &id in LANGUAGES {
        if let Some(error_rate) = parse_error_rate(id, text)
            && error_rate <= 0.6
        {
            let kw = keyword_hits(id, text);
            let score = kw as f64 * 2.0 + (1.0 - error_rate);
            candidates.push((id, score));
        }
    }

    candidates
        .into_iter()
        .max_by(|a, b| crate::scoring::cmp_f64_asc(a.1, b.1))
        .map(|(id, _)| id)
}

fn looks_like_code(text: &str) -> bool {
    let code_symbols = [
        '{', '}', '(', ')', ';', '=', '|', '&', '+', '-', '*', '/', '<', '>',
    ];
    text.chars().filter(|c| code_symbols.contains(c)).count() >= 2
}

fn keyword_hits(id: LanguageId, text: &str) -> usize {
    let lower = text.to_ascii_lowercase();
    let keywords: &[&str] = match id {
        LanguageId::Rust => &[
            "fn ",
            "let ",
            "mut ",
            "impl ",
            "struct ",
            "enum ",
            "use ",
            "pub ",
            "match ",
            "crate",
            "unsafe",
            "macro_rules!",
        ],
        LanguageId::Python => &[
            "def ",
            "class ",
            "import ",
            "from ",
            "return ",
            "if __name__",
            "lambda",
            "none",
            "true",
            "false",
        ],
        LanguageId::JavaScript => &[
            "function ",
            "const ",
            "let ",
            "var ",
            "=>",
            "undefined",
            "null",
            "document.",
            "window.",
        ],
        LanguageId::TypeScript => &[
            "interface ",
            "type ",
            "readonly ",
            "as ",
            "extends ",
            "implements ",
            "namespace ",
            "enum ",
        ],
        LanguageId::Go => &[
            "func ",
            "package ",
            "import ",
            ":=",
            "chan ",
            "defer",
            "goroutine",
        ],
        LanguageId::Java => &[
            "public class",
            "private ",
            "protected ",
            "static ",
            "void ",
            "system.",
            "string[]",
        ],
        LanguageId::C => &["#include", "int main", "printf(", "malloc(", "sizeof("],
        LanguageId::Cpp => &["std::", "template<", "namespace ", "cout", "::", "nullptr"],
    };
    keywords.iter().filter(|k| lower.contains(**k)).count()
}

fn parse_error_rate(id: LanguageId, text: &str) -> Option<f64> {
    let mut parser = Parser::new();
    parser.set_language(&language(id)).ok()?;
    let tree = parser.parse(text, None)?;
    let root = tree.root_node();
    let total = root.descendant_count().max(1);
    let errors = count_errors(root);
    Some(errors as f64 / total as f64)
}

fn count_errors(node: tree_sitter::Node<'_>) -> usize {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_rust() {
        let code = r#"use std::io;

fn main() {
    println!(\"hello\");
    let x = 42;
    if x > 0 {
        println!(\"positive\");
    }
}

struct Point {
    x: f64,
    y: f64,
}
"#;
        assert_eq!(detect_language(code), Some(LanguageId::Rust));
    }

    #[test]
    fn detects_python() {
        let code = r#"import os

def foo():
    return 42

class Bar:
    def __init__(self):
        self.value = 0

if __name__ == \"__main__\":
    print(foo())
"#;
        assert_eq!(detect_language(code), Some(LanguageId::Python));
    }

    #[test]
    fn detects_typescript() {
        let code = "interface User { id: number; name: string; }";
        assert_eq!(detect_language(code), Some(LanguageId::TypeScript));
    }

    #[test]
    fn prose_not_detected() {
        let text = "The quick brown fox jumps over the lazy dog. This is plain English prose.";
        assert_eq!(detect_language(text), None);
    }
}
