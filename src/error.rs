/// Crate-level error type for the giggle audio analysis library.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Invalid parameter value.
    #[error("invalid parameter `{name}`: got {value}, {reason}")]
    InvalidParameter {
        name: &'static str,
        value: String,
        reason: String,
    },

    /// Audio data is empty when a non-empty signal was required.
    #[error("audio data is empty")]
    EmptyAudio,

    /// Audio data contains non-finite values (NaN or Inf).
    #[error("audio data contains non-finite values")]
    NonFiniteAudio,

    /// Input array has incorrect shape for the operation.
    #[error("shape mismatch: expected {expected}, got {got}")]
    ShapeMismatch { expected: String, got: String },

    /// A required dimension is zero or invalid.
    #[error("invalid size for `{name}`: {value} ({reason})")]
    InvalidSize {
        name: &'static str,
        value: usize,
        reason: &'static str,
    },

    /// Frequency range is invalid.
    #[error("invalid frequency range: fmin={fmin}, fmax={fmax} ({reason})")]
    InvalidFrequencyRange {
        fmin: f32,
        fmax: f32,
        reason: String,
    },

    /// Comparison failed due to NaN values.
    #[error("comparison failed: data contains NaN values")]
    NanComparison,

    /// Audio I/O errors.
    #[error(transparent)]
    Audio(#[from] crate::io::AudioError),

    /// File I/O errors.
    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// Validation error (legacy string-based).
    #[error("{0}")]
    Validation(String),
}

impl From<String> for Error {
    fn from(s: String) -> Self {
        Error::Validation(s)
    }
}

/// Convenience Result type for giggle operations.
pub type Result<T> = std::result::Result<T, Error>;
