//! Document parsing and chunking for project indexing.
//!
//! Supports:
//! - AST-aware code chunking via tree-sitter (Rust, Python, JS/TS, Go, Java, C/C++).
//! - Markdown text extraction.
//! - HTML text extraction.

use std::path::Path;
use tree_sitter::{Language, Node, Parser};

pub mod chunking;
pub mod graph;
pub mod impact;

/// A chunk of source or documentation ready for embedding.
#[derive(Debug, Clone)]
pub struct Chunk {
    pub id: String,
    pub source: String,
    pub kind: ChunkKind,
    pub language: Option<String>,
    pub text: String,
    pub summary: Option<String>,
    pub start_line: usize,
    pub end_line: usize,
    /// Optional link to the owning symbol in a [`crate::graph::CodeGraph`].
    pub symbol_id: Option<crate::graph::SymbolId>,
    /// Related symbol ids (callers, callees, implementors) for context expansion.
    pub related_symbols: Vec<crate::graph::SymbolId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkKind {
    Code,
    Doc,
}

impl Chunk {
    /// A short label for the chunk kind, useful for collection routing.
    pub fn kind_str(&self) -> &'static str {
        match self.kind {
            ChunkKind::Code => "code",
            ChunkKind::Doc => "doc",
        }
    }
}

/// Errors during chunking.
#[derive(Debug, thiserror::Error)]
pub enum ChunkError {
    #[error("unsupported language: {0}")]
    UnsupportedLanguage(String),
    #[error("parse error: {0}")]
    Parse(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

fn language_by_extension(path: &Path) -> Option<Language> {
    match path.extension().and_then(|e| e.to_str())? {
        "rs" => Some(tree_sitter_rust::LANGUAGE.into()),
        "py" => Some(tree_sitter_python::LANGUAGE.into()),
        "js" | "jsx" => Some(tree_sitter_javascript::LANGUAGE.into()),
        "ts" | "tsx" => Some(tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into()),
        "go" => Some(tree_sitter_go::LANGUAGE.into()),
        "java" => Some(tree_sitter_java::LANGUAGE.into()),
        "c" | "h" => Some(tree_sitter_c::LANGUAGE.into()),
        "cpp" | "cc" | "hpp" => Some(tree_sitter_cpp::LANGUAGE.into()),
        _ => None,
    }
}

fn language_name(path: &Path) -> Option<String> {
    Some(
        match path.extension().and_then(|e| e.to_str())? {
            "rs" => "rust",
            "py" => "python",
            "js" | "jsx" => "javascript",
            "ts" | "tsx" => "typescript",
            "go" => "go",
            "java" => "java",
            "c" | "h" => "c",
            "cpp" | "cc" | "hpp" => "cpp",
            _ => return None,
        }
        .into(),
    )
}

fn semantic_node_types(language: &str) -> &'static [&'static str] {
    match language {
        "rust" => &[
            "function_item",
            "impl_item",
            "struct_item",
            "enum_item",
            "trait_item",
            "macro_definition",
        ],
        "python" => &[
            "function_definition",
            "class_definition",
            "decorated_definition",
        ],
        "javascript" | "typescript" => &[
            "function_declaration",
            "class_declaration",
            "method_definition",
            "arrow_function",
        ],
        "go" => &[
            "function_declaration",
            "method_declaration",
            "type_declaration",
        ],
        "java" => &[
            "method_declaration",
            "class_declaration",
            "interface_declaration",
        ],
        "c" | "cpp" => &["function_definition", "class_specifier", "struct_specifier"],
        _ => &[],
    }
}

fn first_line(text: &str) -> String {
    text.lines().next().unwrap_or("").trim().to_string()
}

/// Chunk a source code file into symbol-level pieces.
pub fn chunk_code(file_path: &str, source: &str) -> Result<Vec<Chunk>, ChunkError> {
    let path = Path::new(file_path);
    let language = language_name(path)
        .ok_or_else(|| ChunkError::UnsupportedLanguage(file_path.to_string()))?;
    let grammar = language_by_extension(path)
        .ok_or_else(|| ChunkError::UnsupportedLanguage(language.clone()))?;

    let mut parser = Parser::new();
    parser
        .set_language(&grammar)
        .map_err(|e| ChunkError::Parse(e.to_string()))?;
    let tree = parser
        .parse(source, None)
        .ok_or_else(|| ChunkError::Parse("failed to parse".into()))?;

    let root = tree.root_node();
    let targets = semantic_node_types(&language);
    let mut chunks = Vec::new();
    let mut idx = 0usize;

    fn collect_nodes<'a>(node: Node<'a>, targets: &[&str], out: &mut Vec<Node<'a>>) {
        if targets.contains(&node.kind()) {
            out.push(node);
        }
        for i in 0..node.child_count() {
            collect_nodes(node.child(i).expect("child exists"), targets, out);
        }
    }

    let mut nodes = Vec::new();
    collect_nodes(root, targets, &mut nodes);

    for node in nodes {
        let text = source[node.start_byte()..node.end_byte()].to_string();
        if text.trim().is_empty() {
            continue;
        }
        chunks.push(Chunk {
            id: format!("{}-{}", file_path, idx),
            source: file_path.into(),
            kind: ChunkKind::Code,
            language: Some(language.clone()),
            summary: Some(first_line(&text)),
            start_line: node.start_position().row + 1,
            end_line: node.end_position().row + 1,
            text,
            symbol_id: None,
            related_symbols: Vec::new(),
        });
        idx += 1;
    }

    Ok(chunks)
}

/// Extract plain text from Markdown, preserving headings as structure hints.
pub fn parse_markdown(file_path: &str, source: &str) -> Vec<Chunk> {
    let mut text = String::new();
    let parser = pulldown_cmark::Parser::new(source);
    for event in parser {
        use pulldown_cmark::Event;
        match event {
            Event::Text(t) | Event::Code(t) => text.push_str(&t),
            Event::SoftBreak | Event::HardBreak => text.push('\n'),
            Event::Html(t) => text.push_str(&t),
            _ => {}
        }
    }

    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }

    vec![Chunk {
        id: format!("{}-0", file_path),
        source: file_path.into(),
        kind: ChunkKind::Doc,
        language: Some("markdown".into()),
        summary: Some(first_line(trimmed)),
        start_line: 1,
        end_line: source.lines().count().max(1),
        text: trimmed.into(),
        symbol_id: None,
        related_symbols: Vec::new(),
    }]
}

/// Extract plain text from HTML.
pub fn parse_html(file_path: &str, source: &str) -> Vec<Chunk> {
    let document = scraper::Html::parse_document(source);
    let text = document.root_element().text().collect::<String>();
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }

    vec![Chunk {
        id: format!("{}-0", file_path),
        source: file_path.into(),
        kind: ChunkKind::Doc,
        language: Some("html".into()),
        summary: Some(first_line(trimmed)),
        start_line: 1,
        end_line: source.lines().count().max(1),
        text: trimmed.into(),
        symbol_id: None,
        related_symbols: Vec::new(),
    }]
}

/// Parse a documentation file based on extension.
pub fn parse_doc(file_path: &str, source: &str) -> Vec<Chunk> {
    match Path::new(file_path).extension().and_then(|e| e.to_str()) {
        Some("md") => parse_markdown(file_path, source),
        Some("html") | Some("htm") => parse_html(file_path, source),
        _ => Vec::new(),
    }
}

/// Walk a project directory respecting `.gitignore` and return readable file paths.
pub fn walk_project(project_root: &Path) -> Result<Vec<String>, ChunkError> {
    use ignore::WalkBuilder;

    let mut paths = Vec::new();
    let walker = WalkBuilder::new(project_root)
        .hidden(false)
        .git_ignore(true)
        .git_global(false)
        .git_exclude(true)
        .build();

    for entry in walker {
        let entry = entry.map_err(|e| {
            ChunkError::Io(
                e.into_io_error()
                    .unwrap_or_else(|| std::io::Error::other("walk error")),
            )
        })?;
        if entry.file_type().map(|ft| ft.is_file()).unwrap_or(false) {
            paths.push(entry.path().to_string_lossy().to_string());
        }
    }
    Ok(paths)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunker_respects_gitignore() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::create_dir(root.join(".git")).unwrap();
        std::fs::write(root.join("tracked.txt"), "tracked").unwrap();
        std::fs::write(root.join(".gitignore"), "ignored.txt\n").unwrap();
        std::fs::write(root.join("ignored.txt"), "ignored").unwrap();

        let paths = walk_project(root).unwrap();
        let names: Vec<_> = paths
            .iter()
            .map(|p| {
                Path::new(p)
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
                    .to_string()
            })
            .collect();
        assert!(names.contains(&"tracked.txt".into()));
        assert!(!names.contains(&"ignored.txt".into()));
    }

    #[test]
    fn chunker_chunks_rust_function_by_symbol() {
        let source = r#"
fn add(a: i32, b: i32) -> i32 {
    a + b
}

struct Point {
    x: f64,
    y: f64,
}
"#;
        let chunks = chunk_code("test.rs", source).unwrap();
        assert!(!chunks.is_empty());
        assert!(chunks.iter().any(|c| c.text.contains("fn add")));
        assert!(chunks.iter().any(|c| c.text.contains("struct Point")));
        for c in &chunks {
            assert_eq!(c.language.as_deref(), Some("rust"));
            assert!(c.summary.as_ref().unwrap().len() <= c.text.lines().next().unwrap().len());
        }
    }
}
