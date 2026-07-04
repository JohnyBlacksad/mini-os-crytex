use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// Errors that can occur when accessing a [`CcrStore`].
#[derive(Debug, thiserror::Error)]
pub enum CcrStoreError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("lock poisoned")]
    LockPoisoned,
}

/// A simple Compress-Cache-Retrieve store.
///
/// When a compressor offloads content, it stores the original under a stable
/// key and emits a short retrieval marker instead. The original can be fetched
/// later by the same key.
pub trait CcrStore: Send + Sync {
    /// Store `content` under `key`. Overwrites any existing entry.
    fn put(&self, key: &str, content: String) -> Result<(), CcrStoreError>;

    /// Retrieve content by `key`. Returns `None` if the key is unknown.
    fn get(&self, key: &str) -> Result<Option<String>, CcrStoreError>;
}

/// Parse a CCR retrieval marker of the form `[original diff stored: ccr:<key> ...]`
/// and return the key, if present.
pub fn parse_marker(text: &str) -> Option<&str> {
    let start = text.find("ccr:")?;
    let key_start = start + 4;
    let key_end = text[key_start..]
        .find(|c: char| c.is_whitespace() || c == ']')
        .map(|i| key_start + i)
        .unwrap_or(text.len());
    Some(&text[key_start..key_end])
}

/// Retrieve original content from a store using a retrieval marker embedded in
/// compressed text.
pub fn retrieve(store: &dyn CcrStore, marker: &str) -> Result<Option<String>, CcrStoreError> {
    match parse_marker(marker) {
        Some(key) => store.get(key),
        None => Ok(None),
    }
}

/// In-memory CCR store. Useful for local / single-process deployments.
#[derive(Debug, Clone, Default)]
pub struct InMemoryCcrStore {
    data: Arc<Mutex<HashMap<String, String>>>,
}

impl InMemoryCcrStore {
    pub fn new() -> Self {
        Self::default()
    }
}

impl CcrStore for InMemoryCcrStore {
    fn put(&self, key: &str, content: String) -> Result<(), CcrStoreError> {
        self.data
            .lock()
            .map_err(|_| CcrStoreError::LockPoisoned)?
            .insert(key.to_string(), content);
        Ok(())
    }

    fn get(&self, key: &str) -> Result<Option<String>, CcrStoreError> {
        Ok(self
            .data
            .lock()
            .map_err(|_| CcrStoreError::LockPoisoned)?
            .get(key)
            .cloned())
    }
}

/// Persistent on-disk CCR store.
///
/// Each entry is written to its own file under `root` using the key as the
/// file name. Writes are atomic (temp file + rename) so the cache stays
/// consistent even if the process crashes.
#[derive(Debug, Clone)]
pub struct DiskCcrStore {
    root: PathBuf,
}

impl DiskCcrStore {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    fn path_for(&self, key: &str) -> PathBuf {
        // Keys are hex hashes, so safe for filenames. Still sanitize for safety.
        let safe: String = key
            .chars()
            .filter(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_')
            .collect();
        self.root.join(format!("{}.ccr", safe))
    }
}

impl CcrStore for DiskCcrStore {
    fn put(&self, key: &str, content: String) -> Result<(), CcrStoreError> {
        std::fs::create_dir_all(&self.root)?;
        let path = self.path_for(key);
        let temp = path.with_extension("tmp");
        atomic_write(&path, &temp, content)?;
        Ok(())
    }

    fn get(&self, key: &str) -> Result<Option<String>, CcrStoreError> {
        let path = self.path_for(key);
        match std::fs::read_to_string(&path) {
            Ok(text) => Ok(Some(text)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e.into()),
        }
    }
}

#[cfg(not(target_os = "windows"))]
fn atomic_write(final_path: &Path, temp_path: &Path, content: String) -> std::io::Result<()> {
    std::fs::write(temp_path, content)?;
    std::fs::rename(temp_path, final_path)?;
    Ok(())
}

#[cfg(target_os = "windows")]
fn atomic_write(final_path: &Path, temp_path: &Path, content: String) -> std::io::Result<()> {
    // On Windows, rename over an existing file is not atomic, so remove first.
    std::fs::write(temp_path, content)?;
    let _ = std::fs::remove_file(final_path);
    std::fs::rename(temp_path, final_path)?;
    Ok(())
}

/// Compute a short, deterministic cache key for `content`.
pub fn compute_key(content: &str) -> String {
    let hash = blake3::hash(content.as_bytes());
    hash.to_hex()[..24].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn in_memory_store_round_trips() {
        let store = InMemoryCcrStore::new();
        store.put("abc", "original".to_string()).unwrap();
        assert_eq!(store.get("abc").unwrap(), Some("original".to_string()));
        assert_eq!(store.get("missing").unwrap(), None);
    }

    #[test]
    fn disk_store_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let store = DiskCcrStore::new(dir.path());
        store.put("abc123", "original diff".to_string()).unwrap();
        assert_eq!(
            store.get("abc123").unwrap(),
            Some("original diff".to_string())
        );
        assert_eq!(store.get("missing").unwrap(), None);
    }

    #[test]
    fn disk_store_overwrites() {
        let dir = tempfile::tempdir().unwrap();
        let store = DiskCcrStore::new(dir.path());
        store.put("key", "first".to_string()).unwrap();
        store.put("key", "second".to_string()).unwrap();
        assert_eq!(store.get("key").unwrap(), Some("second".to_string()));
    }

    #[test]
    fn parse_marker_extracts_key() {
        let text = "[original diff stored: ccr:abc123 (retrieve if needed)]";
        assert_eq!(parse_marker(text), Some("abc123"));
    }

    #[test]
    fn retrieve_round_trips_via_marker() {
        let store = InMemoryCcrStore::new();
        store.put("k1", "original".to_string()).unwrap();
        let marker = "compressed text\n[original diff stored: ccr:k1]";
        assert_eq!(
            retrieve(&store, marker).unwrap(),
            Some("original".to_string())
        );
    }

    #[test]
    fn key_is_deterministic_and_short() {
        let k1 = compute_key("hello world");
        let k2 = compute_key("hello world");
        assert_eq!(k1, k2);
        assert_eq!(k1.len(), 24);
    }
}
