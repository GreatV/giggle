//! File utility functions (equivalent to librosa.util.files).
//!
//! Provides audio file discovery and library metadata utilities.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// Default audio file extensions recognised by [`find_files`].
pub const DEFAULT_AUDIO_EXTENSIONS: &[&str] = &["aac", "au", "flac", "m4a", "mp3", "ogg", "wav"];

/// Get a sorted list of audio files in a directory or directory sub-tree.
///
/// # Arguments
/// * `directory` - Path to look for files
/// * `ext` - File extensions to include. `None` uses the default set
///   (`aac`, `au`, `flac`, `m4a`, `mp3`, `ogg`, `wav`).
/// * `recurse` - If `true`, search all subfolders of `directory`
/// * `case_sensitive` - If `false`, both lower- and upper-case extensions match
/// * `limit` - Return at most this many files. `None` returns all
/// * `offset` - Skip the first `offset` files (negative counts from the end)
///
/// # Returns
/// A sorted list of matching file paths.
///
/// # Examples
/// ```no_run
/// use giggle::files::find_files;
///
/// // All audio files under ~/Music
/// let files = find_files("~/Music", None, true, false, None, 0);
///
/// // Only mp3 and ogg, non-recursive
/// let files = find_files("./samples", Some(&["mp3", "ogg"]), false, false, None, 0);
///
/// // First 10 wav files
/// let files = find_files("./data", Some(&["wav"]), true, false, Some(10), 0);
/// ```
pub fn find_files(
    directory: &str,
    ext: Option<&[&str]>,
    recurse: bool,
    case_sensitive: bool,
    limit: Option<usize>,
    offset: i64,
) -> Vec<PathBuf> {
    let default_ext;
    let extensions: &[&str] = match ext {
        Some(e) => e,
        None => {
            default_ext = DEFAULT_AUDIO_EXTENSIONS;
            default_ext
        }
    };

    // Build the set of extensions to match
    let mut ext_set = BTreeSet::new();
    for e in extensions {
        let lower = e.to_lowercase();
        ext_set.insert(lower.clone());
        if !case_sensitive {
            ext_set.insert(lower.to_uppercase());
        }
    }

    // Expand ~ to home directory
    let dir_path = expand_tilde(directory);

    // Collect matching files
    let mut files = BTreeSet::new();
    if recurse {
        collect_files_recursive(&dir_path, &ext_set, &mut files);
    } else {
        collect_files_flat(&dir_path, &ext_set, &mut files);
    }

    // BTreeSet is already sorted; convert to Vec
    let mut result: Vec<PathBuf> = files.into_iter().collect();

    // Apply offset
    let start = if offset >= 0 {
        offset as usize
    } else {
        result.len().saturating_sub((-offset) as usize)
    };

    if start >= result.len() {
        return Vec::new();
    }
    result = result.split_off(start);

    // Apply limit
    if let Some(lim) = limit {
        result.truncate(lim);
    }

    result
}

/// Return the version string of this crate.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Return citation information for giggle.
///
/// # Arguments
/// * `version` - Optional version string. If `None`, uses the current version.
///
/// # Returns
/// A citation string for the given version.
///
/// # Examples
/// ```
/// let citation = giggle::files::cite(None);
/// assert!(citation.contains("giggle"));
/// ```
pub fn cite(ver: Option<&str>) -> String {
    let v = ver.unwrap_or(env!("CARGO_PKG_VERSION"));
    format!(
        "giggle v{v} — a pure-Rust audio analysis library (port of librosa). \
         https://github.com/user/giggle"
    )
}

/// Metadata for a bundled example track.
#[derive(Debug, Clone)]
pub struct ExampleInfo {
    /// Short key used to identify the example (e.g. "trumpet").
    pub key: &'static str,
    /// Human-readable description.
    pub desc: &'static str,
    /// Relative filename inside the examples directory.
    pub filename: &'static str,
}

/// Built-in example catalogue.
///
/// Extend this array as example audio files are added to the project.
static EXAMPLES: &[ExampleInfo] = &[
    // Placeholder — add real entries when example audio files are bundled
];

/// List all available example recordings.
///
/// # Returns
/// A slice of [`ExampleInfo`] describing each available example.
///
/// # Examples
/// ```
/// let examples = giggle::files::list_examples();
/// for ex in examples {
///     println!("{:10}\t{}", ex.key, ex.desc);
/// }
/// ```
pub fn list_examples() -> &'static [ExampleInfo] {
    EXAMPLES
}

/// Look up info for a specific example by key.
///
/// # Returns
/// `Some(&ExampleInfo)` if found, `None` otherwise.
pub fn example_info(key: &str) -> Option<&'static ExampleInfo> {
    EXAMPLES.iter().find(|e| e.key == key)
}

/// Retrieve the filesystem path for a bundled example recording.
///
/// Looks for the file relative to the `examples/` directory at the
/// crate root. Returns `None` if the key is unknown or the file
/// does not exist on disk.
///
/// # Arguments
/// * `key` - The identifier for the example track
///
/// # Returns
/// The absolute path to the example file, or `None`.
pub fn example(key: &str) -> Option<PathBuf> {
    let info = example_info(key)?;
    // Try to locate relative to CARGO_MANIFEST_DIR at compile time,
    // falling back to ./examples/ at runtime.
    let base = option_env!("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    let path = base.join("examples").join(info.filename);
    if path.is_file() { Some(path) } else { None }
}

// ─── internal helpers ────────────────────────────────────────────────

fn expand_tilde(path: &str) -> PathBuf {
    if let Some(rest) = path.strip_prefix("~/")
        && let Some(home) = std::env::var_os("HOME")
    {
        return PathBuf::from(home).join(rest);
    }
    PathBuf::from(path)
}

fn matches_extension(path: &Path, ext_set: &BTreeSet<String>) -> bool {
    match path.extension().and_then(|e| e.to_str()) {
        Some(e) => ext_set.contains(e),
        None => false,
    }
}

fn collect_files_flat(dir: &Path, ext_set: &BTreeSet<String>, out: &mut BTreeSet<PathBuf>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_file() && matches_extension(&path, ext_set) {
            out.insert(path);
        }
    }
}

fn collect_files_recursive(dir: &Path, ext_set: &BTreeSet<String>, out: &mut BTreeSet<PathBuf>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_files_recursive(&path, ext_set, out);
        } else if path.is_file() && matches_extension(&path, ext_set) {
            out.insert(path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_find_files_basic() {
        let tmp = std::env::temp_dir().join("giggle_find_files_test");
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(tmp.join("sub")).unwrap();

        fs::write(tmp.join("a.wav"), b"").unwrap();
        fs::write(tmp.join("b.mp3"), b"").unwrap();
        fs::write(tmp.join("c.txt"), b"").unwrap();
        fs::write(tmp.join("sub").join("d.flac"), b"").unwrap();

        // Recursive, default extensions
        let files = find_files(tmp.to_str().unwrap(), None, true, false, None, 0);
        assert_eq!(files.len(), 3); // wav, mp3, flac — not txt

        // Non-recursive
        let files = find_files(tmp.to_str().unwrap(), None, false, false, None, 0);
        assert_eq!(files.len(), 2); // wav, mp3

        // Filter by ext
        let files = find_files(tmp.to_str().unwrap(), Some(&["wav"]), true, false, None, 0);
        assert_eq!(files.len(), 1);

        // Limit
        let files = find_files(tmp.to_str().unwrap(), None, true, false, Some(2), 0);
        assert_eq!(files.len(), 2);

        // Negative offset (last 1)
        let files = find_files(tmp.to_str().unwrap(), None, true, false, None, -1);
        assert_eq!(files.len(), 1);

        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_find_files_case_insensitive() {
        let tmp = std::env::temp_dir().join("giggle_find_files_case_test");
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();

        fs::write(tmp.join("a.WAV"), b"").unwrap();
        fs::write(tmp.join("b.wav"), b"").unwrap();

        let files = find_files(tmp.to_str().unwrap(), Some(&["wav"]), false, false, None, 0);
        assert_eq!(files.len(), 2);

        let files = find_files(tmp.to_str().unwrap(), Some(&["wav"]), false, true, None, 0);
        assert_eq!(files.len(), 1); // only b.wav

        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_cite() {
        let c = cite(None);
        assert!(c.contains("giggle"));
        assert!(c.contains(env!("CARGO_PKG_VERSION")));
    }

    #[test]
    fn test_list_examples() {
        let _ = list_examples(); // should not panic
    }

    #[test]
    fn test_version() {
        assert!(!version().is_empty());
    }
}
