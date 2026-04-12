//! Disk-persisted AST cache.
//!
//! Saves canonicalised expression trees to
//! `$ELWORTHY_CACHE_DIR` (or `$XDG_CACHE_HOME/elworthy/`, or
//! `~/.cache/elworthy/`) keyed on the structural hash of the `Expr`. The
//! machine code from Cranelift embeds mmap addresses and libm symbol
//! resolutions that are process-local, so we persist the AST and
//! recompile on warm-start. Recompilation is milliseconds and the AST is
//! the source of truth.
//!
//! The cache directory is intentionally outside the repo: it is a
//! runtime artefact, never tracked by git.

use crate::hash::expr_hash;
use crate::serial;
use elworthy_expr::Expr;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// A format-version stamp embedded in each cache file. Bump this whenever
/// the `Expr` enum gains a new node type so stale files fail loudly.
const FORMAT_VERSION: u32 = 1;

/// Disk-backed AST cache. Files live at `<root>/<hex-hash>.ast` and are
/// self-validating via the embedded format version.
pub struct DiskCache {
    root: PathBuf,
}

impl DiskCache {
    /// Open the cache rooted at `path`, creating the directory if it does
    /// not exist.
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let root = path.as_ref().to_path_buf();
        fs::create_dir_all(&root)?;
        Ok(Self { root })
    }

    /// Open the cache at the default location: `$ELWORTHY_CACHE_DIR`,
    /// then `$XDG_CACHE_HOME/elworthy/`, then `~/.cache/elworthy/`.
    pub fn open_default() -> io::Result<Self> {
        let root = default_cache_dir()?;
        Self::open(root)
    }

    /// Return the on-disk path for an expression, whether or not it
    /// exists.
    pub fn path_for(&self, expr: &Expr) -> PathBuf {
        self.root.join(format!("{:016x}.ast", expr_hash(expr)))
    }

    /// Persist an expression. Overwrites any prior entry for the same
    /// structural hash.
    pub fn store(&self, expr: &Expr) -> io::Result<()> {
        let path = self.path_for(expr);
        let tmp = path.with_extension("ast.tmp");
        {
            let mut f = fs::File::create(&tmp)?;
            io::Write::write_all(&mut f, &FORMAT_VERSION.to_le_bytes())?;
            serial::write_expr(&mut f, expr)?;
        }
        fs::rename(tmp, path)?;
        Ok(())
    }

    /// Load an expression whose hash matches `target_hash`. Returns
    /// `Ok(None)` for a miss, `Err` only on IO or format errors.
    pub fn load_by_hash(&self, target_hash: u64) -> io::Result<Option<Expr>> {
        let path = self.root.join(format!("{target_hash:016x}.ast"));
        if !path.exists() {
            return Ok(None);
        }
        let mut f = fs::File::open(path)?;
        let mut ver = [0u8; 4];
        io::Read::read_exact(&mut f, &mut ver)?;
        if u32::from_le_bytes(ver) != FORMAT_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "stale elworthy disk-cache format version",
            ));
        }
        let expr = serial::read_expr(&mut f)?;
        Ok(Some(expr))
    }

    /// Remove every `.ast` file under the cache root.
    pub fn clear(&self) -> io::Result<()> {
        for entry in fs::read_dir(&self.root)? {
            let entry = entry?;
            if entry.path().extension().and_then(|s| s.to_str()) == Some("ast") {
                fs::remove_file(entry.path())?;
            }
        }
        Ok(())
    }

    pub fn root(&self) -> &Path {
        &self.root
    }
}

fn default_cache_dir() -> io::Result<PathBuf> {
    if let Ok(p) = std::env::var("ELWORTHY_CACHE_DIR") {
        return Ok(PathBuf::from(p));
    }
    if let Ok(p) = std::env::var("XDG_CACHE_HOME") {
        return Ok(PathBuf::from(p).join("elworthy"));
    }
    if let Ok(home) = std::env::var("HOME") {
        return Ok(PathBuf::from(home).join(".cache/elworthy"));
    }
    Err(io::Error::new(
        io::ErrorKind::NotFound,
        "no ELWORTHY_CACHE_DIR, XDG_CACHE_HOME, or HOME set",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::expr_hash;

    fn tempdir() -> PathBuf {
        let base = std::env::temp_dir().join(format!(
            "elworthy-diskcache-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&base).unwrap();
        base
    }

    #[test]
    fn store_then_load_roundtrips() {
        let dir = tempdir();
        let cache = DiskCache::open(&dir).unwrap();
        let e = Expr::param(0) * Expr::state(0) + Expr::c(1.0);
        cache.store(&e).unwrap();
        let loaded = cache.load_by_hash(expr_hash(&e)).unwrap();
        assert!(loaded.is_some());
        let loaded = loaded.unwrap();
        assert_eq!(expr_hash(&e), expr_hash(&loaded));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn miss_returns_none() {
        let dir = tempdir();
        let cache = DiskCache::open(&dir).unwrap();
        let e = Expr::c(42.0);
        assert!(cache.load_by_hash(expr_hash(&e)).unwrap().is_none());
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn clear_removes_files() {
        let dir = tempdir();
        let cache = DiskCache::open(&dir).unwrap();
        let e = Expr::state(0);
        cache.store(&e).unwrap();
        cache.clear().unwrap();
        assert!(cache.load_by_hash(expr_hash(&e)).unwrap().is_none());
        let _ = fs::remove_dir_all(dir);
    }
}
