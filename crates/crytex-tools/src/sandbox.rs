use std::path::{Component, Path, PathBuf};

/// A root that the sandbox can expose to a tool.
#[derive(Debug, Clone)]
pub enum SandboxRoot {
    /// Project workspace. Writes allowed only if `WRITE` capability is granted.
    Project,
    /// Global crytex config/data directory (read-only by default).
    Global,
    /// Explicit host path exposed for a specific tool invocation.
    Host(PathBuf),
}

/// Errors originating from the path sandbox.
#[derive(Debug, thiserror::Error)]
pub enum SandboxError {
    #[error("path escapes allowed roots: {0}")]
    PathEscape(String),
    #[error("path is not within any allowed root: {0}")]
    OutsideRoot(String),
    #[error("symbolic link points outside allowed roots: {0}")]
    SymlinkEscape(String),
    #[error("operation not allowed on path: {0}")]
    NotAllowed(String),
}

/// Validates that all filesystem access stays inside allowed roots.
#[derive(Debug, Clone)]
pub struct PathSandbox {
    project_root: PathBuf,
    global_root: PathBuf,
    blocked_paths: Vec<PathBuf>,
}

impl PathSandbox {
    pub fn new(project_root: impl AsRef<Path>, global_root: impl AsRef<Path>) -> Self {
        let blocked_paths: Vec<PathBuf> = Self::default_blocked_paths()
            .into_iter()
            .filter_map(|p| std::fs::canonicalize(&p).ok())
            .collect();
        Self {
            project_root: project_root.as_ref().to_path_buf(),
            global_root: global_root.as_ref().to_path_buf(),
            blocked_paths,
        }
    }

    #[cfg(test)]
    fn new_with_blocked(
        project_root: impl AsRef<Path>,
        global_root: impl AsRef<Path>,
        blocked_paths: Vec<PathBuf>,
    ) -> Self {
        Self {
            project_root: project_root.as_ref().to_path_buf(),
            global_root: global_root.as_ref().to_path_buf(),
            blocked_paths,
        }
    }

    fn default_blocked_paths() -> Vec<PathBuf> {
        let mut blocked = Vec::new();

        if let Some(home) = home_dir() {
            for rel in [".ssh", ".env", ".aws", ".kube", ".gnupg"] {
                blocked.push(home.join(rel));
            }
        }

        #[cfg(unix)]
        {
            blocked.extend(
                ["/etc", "/root", "/proc", "/sys", "/dev"]
                    .iter()
                    .map(PathBuf::from),
            );
        }

        #[cfg(windows)]
        {
            if let Some(windir) =
                std::env::var_os("WINDIR").or_else(|| std::env::var_os("SystemRoot"))
            {
                blocked.push(PathBuf::from(windir));
            }
            if let Some(pf) = std::env::var_os("ProgramFiles") {
                blocked.push(PathBuf::from(pf));
            }
            if let Some(pf86) = std::env::var_os("ProgramFiles(x86)") {
                blocked.push(PathBuf::from(pf86));
            }
        }

        blocked
    }

    /// Resolve a user-supplied path against the project root.
    ///
    /// Rejects absolute paths, `..` traversal, symlinks that escape the root,
    /// and paths that hit the sensitive-path blocklist.
    pub fn resolve(&self, raw: impl AsRef<Path>) -> Result<PathBuf, SandboxError> {
        let raw = raw.as_ref();
        let normalized = lexical_normalize(raw)?;
        let joined = self.project_root.join(&normalized);
        self.check_within(&joined, &self.project_root, &normalized)
    }

    /// Resolve a path that may target the global crytex directory (read-only).
    pub fn resolve_global(&self, raw: impl AsRef<Path>) -> Result<PathBuf, SandboxError> {
        let raw = raw.as_ref();
        let normalized = lexical_normalize(raw)?;
        let joined = self.global_root.join(&normalized);
        self.check_within(&joined, &self.global_root, &normalized)
    }

    fn check_within(
        &self,
        joined: &Path,
        root: &Path,
        normalized: &Path,
    ) -> Result<PathBuf, SandboxError> {
        let root_canonical = std::fs::canonicalize(root).unwrap_or_else(|_| root.to_path_buf());
        let real = Self::canonicalize_existing_prefix(joined, &root_canonical)?;

        if !real.starts_with(&root_canonical) {
            if Self::detect_symlink_escape(joined, &root_canonical) {
                return Err(SandboxError::SymlinkEscape(format!(
                    "{} resolves outside {}",
                    joined.display(),
                    root_canonical.display()
                )));
            }
            return Err(SandboxError::PathEscape(format!(
                "{} is outside {}",
                real.display(),
                root_canonical.display()
            )));
        }

        if self.is_blocked(&real) {
            return Err(SandboxError::NotAllowed(format!(
                "path is blocked: {}",
                real.display()
            )));
        }

        // Preserve the caller-friendly normalized path when the file does not
        // yet exist, but anchored to the real root.
        if !joined.exists() {
            Ok(root_canonical.join(normalized))
        } else {
            Ok(real)
        }
    }

    fn canonicalize_existing_prefix(path: &Path, root: &Path) -> Result<PathBuf, SandboxError> {
        if path.exists() {
            return std::fs::canonicalize(path).map_err(|e| {
                SandboxError::PathEscape(format!(
                    "failed to canonicalize {}: {}",
                    path.display(),
                    e
                ))
            });
        }

        // Walk up until we find an existing directory, canonicalize it to
        // resolve symlinks, then append the non-existing tail.
        let mut existing = path.to_path_buf();
        let mut tail: Vec<std::ffi::OsString> = Vec::new();
        while !existing.exists() {
            if let Some(name) = existing.file_name() {
                tail.push(name.to_os_string());
                existing.pop();
            } else {
                break;
            }
        }
        if existing.as_os_str().is_empty() {
            existing = root.to_path_buf();
        }

        let real_existing = if existing.exists() {
            std::fs::canonicalize(&existing).map_err(|e| {
                SandboxError::PathEscape(format!(
                    "failed to canonicalize {}: {}",
                    existing.display(),
                    e
                ))
            })?
        } else {
            existing
        };

        let mut real = real_existing;
        for name in tail.into_iter().rev() {
            real.push(name);
        }
        Ok(real)
    }

    fn detect_symlink_escape(path: &Path, root: &Path) -> bool {
        let relative = path.strip_prefix(root).unwrap_or(path);
        let mut current = root.to_path_buf();
        for comp in relative.components() {
            if let Component::Normal(name) = comp {
                current.push(name);
                if let Ok(meta) = std::fs::symlink_metadata(&current)
                    && meta.file_type().is_symlink()
                    && let Ok(target) = std::fs::read_link(&current)
                {
                    let resolved = if target.is_absolute() {
                        target
                    } else {
                        current.parent().unwrap_or(root).join(target)
                    };
                    if let Ok(canonical) = std::fs::canonicalize(&resolved)
                        && !canonical.starts_with(root)
                    {
                        return true;
                    }
                }
            }
        }
        false
    }

    fn is_blocked(&self, path: &Path) -> bool {
        self.blocked_paths.iter().any(|b| {
            // Treat an empty blocklist entry as a no-op.
            if b.as_os_str().is_empty() {
                return false;
            }
            path == b || path.starts_with(b)
        })
    }

    /// List children of a directory inside the project root.
    pub fn list_dir(&self, raw: impl AsRef<Path>) -> Result<Vec<PathBuf>, SandboxError> {
        let path = self.resolve(raw)?;
        let mut entries = Vec::new();
        for entry in std::fs::read_dir(&path)? {
            let entry = entry?;
            entries.push(entry.path());
        }
        Ok(entries)
    }
}

fn lexical_normalize(raw: &Path) -> Result<PathBuf, SandboxError> {
    if raw.is_absolute() {
        return Err(SandboxError::OutsideRoot(format!(
            "absolute paths are not allowed: {}",
            raw.display()
        )));
    }

    let mut out = PathBuf::new();
    for comp in raw.components() {
        match comp {
            Component::Normal(p) => out.push(p),
            Component::ParentDir => {
                if !out.pop() {
                    return Err(SandboxError::PathEscape(format!(
                        ".. escapes allowed root: {}",
                        raw.display()
                    )));
                }
            }
            Component::CurDir => {}
            Component::RootDir | Component::Prefix(_) => {
                return Err(SandboxError::OutsideRoot(format!(
                    "absolute paths are not allowed: {}",
                    raw.display()
                )));
            }
        }
    }
    Ok(out)
}

fn home_dir() -> Option<PathBuf> {
    #[cfg(target_os = "windows")]
    {
        std::env::var_os("USERPROFILE").map(PathBuf::from)
    }
    #[cfg(not(target_os = "windows"))]
    {
        std::env::var_os("HOME").map(PathBuf::from)
    }
}

impl From<std::io::Error> for SandboxError {
    fn from(e: std::io::Error) -> Self {
        SandboxError::NotAllowed(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn path_sandbox_rejects_absolute_outside_project() {
        let tmp = tempfile::tempdir().unwrap();
        let sandbox = PathSandbox::new(tmp.path(), "");
        let err = sandbox.resolve("/etc/passwd").unwrap_err();
        assert!(matches!(err, SandboxError::OutsideRoot(_)));
    }

    #[test]
    fn path_sandbox_rejects_dot_dot_escape() {
        let tmp = tempfile::tempdir().unwrap();
        let sandbox = PathSandbox::new(tmp.path(), "");
        let err = sandbox.resolve("a/../../outside.txt").unwrap_err();
        assert!(
            matches!(err, SandboxError::PathEscape(_)),
            "expected PathEscape, got {:?}",
            err
        );
    }

    #[cfg(unix)]
    #[test]
    fn path_sandbox_rejects_symlink_escape() {
        let tmp = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let target = outside.path().join("secret.txt");
        std::fs::write(&target, "secret").unwrap();

        std::os::unix::fs::symlink(&target, tmp.path().join("link")).unwrap();

        let sandbox = PathSandbox::new(tmp.path(), "");
        let err = sandbox.resolve("link").unwrap_err();
        assert!(
            matches!(err, SandboxError::SymlinkEscape(_)),
            "expected SymlinkEscape, got {:?}",
            err
        );
    }

    #[test]
    fn path_sandbox_rejects_blocked_path() {
        let tmp = tempfile::tempdir().unwrap();
        let secret = tmp.path().join("secret");
        std::fs::create_dir(&secret).unwrap();
        let canonical_secret = std::fs::canonicalize(&secret).unwrap();

        let sandbox = PathSandbox::new_with_blocked(tmp.path(), "", vec![canonical_secret]);
        let err = sandbox.resolve("secret/file.txt").unwrap_err();
        assert!(
            matches!(err, SandboxError::NotAllowed(_)),
            "expected NotAllowed, got {:?}",
            err
        );
    }

    #[test]
    fn path_sandbox_allows_normal_relative_path() {
        let tmp = tempfile::tempdir().unwrap();
        let root = std::fs::canonicalize(tmp.path()).unwrap();
        let sandbox = PathSandbox::new(tmp.path(), "");
        let resolved = sandbox.resolve("src/main.rs").unwrap();
        assert!(resolved.starts_with(&root));
    }

    #[test]
    fn path_sandbox_normalizes_dot_components() {
        let tmp = tempfile::tempdir().unwrap();
        let root = std::fs::canonicalize(tmp.path()).unwrap();
        let sandbox = PathSandbox::new(tmp.path(), "");
        let resolved = sandbox.resolve("./src/../lib.rs").unwrap();
        assert!(resolved.ends_with("lib.rs"));
        assert!(resolved.starts_with(&root));
    }
}
