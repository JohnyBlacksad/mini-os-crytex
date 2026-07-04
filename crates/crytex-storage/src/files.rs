use std::path::PathBuf;
use tokio::fs;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("path not found: {0}")]
    NotFound(String),
}

pub struct FileStore {
    root: PathBuf,
}

impl FileStore {
    pub fn new(root: &str) -> Self {
        Self {
            root: PathBuf::from(root),
        }
    }

    pub async fn read(&self, relative: &str) -> Result<Vec<u8>, Error> {
        let path = self.root.join(relative);
        if !path.exists() {
            return Err(Error::NotFound(path.to_string_lossy().to_string()));
        }
        let data = fs::read(&path).await?;
        Ok(data)
    }

    pub async fn write(&self, relative: &str, data: &[u8]) -> Result<(), Error> {
        let path = self.root.join(relative);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).await?;
        }
        fs::write(&path, data).await?;
        Ok(())
    }

    pub async fn exists(&self, relative: &str) -> bool {
        self.root.join(relative).exists()
    }

    pub async fn list(&self, relative: &str) -> Result<Vec<String>, Error> {
        let path = self.root.join(relative);
        let mut entries = Vec::new();
        let mut reader = fs::read_dir(&path).await?;
        while let Some(entry) = reader.next_entry().await? {
            let name = entry.file_name().to_string_lossy().to_string();
            entries.push(name);
        }
        Ok(entries)
    }
}
