use unidiff::PatchSet;

use crate::content::ContentType;

/// Boolean predicate: does `content` parse as a unified diff with real hunks?
pub fn is_diff(content: &str) -> bool {
    if content.is_empty() {
        return false;
    }

    let mut patch = PatchSet::new();
    if patch.parse(content).is_err() {
        return false;
    }

    !patch.is_empty() && patch.files().iter().any(|f| !f.is_empty())
}

/// Typed wrapper returning `ContentType::Diff` on hit.
pub fn detect_diff(content: &str) -> Option<ContentType> {
    if is_diff(content) {
        Some(ContentType::Diff)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_is_not_diff() {
        assert!(!is_diff(""));
        assert_eq!(detect_diff(""), None);
    }

    #[test]
    fn prose_is_not_diff() {
        assert!(!is_diff("The quick brown fox jumps over the lazy dog."));
    }

    #[test]
    fn standard_git_diff_detected() {
        let diff = "diff --git a/foo.py b/foo.py\n\
                    index abc123..def456 100644\n\
                    --- a/foo.py\n\
                    +++ b/foo.py\n\
                    @@ -1,3 +1,4 @@\n\
                     def hello():\n\
                    +    print(\"new\")\n\
                     return \"world\"\n\
                    -    # gone\n";
        assert!(is_diff(diff));
        assert_eq!(detect_diff(diff), Some(ContentType::Diff));
    }

    #[test]
    fn naked_hunk_detected() {
        let diff = "--- a/foo.py\n\
                    +++ b/foo.py\n\
                    @@ -1,2 +1,2 @@\n\
                    -old line\n\
                    +new line\n\
                     context\n";
        assert!(is_diff(diff));
    }
}
