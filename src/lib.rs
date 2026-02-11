//! Audio analysis and music information retrieval library for Rust.
//!
//! Giggle is a comprehensive audio analysis library providing feature parity
//! with Python's [librosa](https://librosa.org/). It covers the full pipeline
//! from audio I/O through spectral analysis, feature extraction, and music
//! information retrieval.
//!
//! # Features
//!
//! - **Spectral analysis** — STFT/ISTFT, CQT/VQT, reassigned spectrogram,
//!   Griffin-Lim phase reconstruction, PCEN, Fast Mellin Transform
//! - **Feature extraction** — mel spectrogram, MFCC, chroma (STFT/CQT/CENS/VQT),
//!   spectral centroid/bandwidth/contrast/rolloff/flatness, tonnetz, tempogram
//! - **Pitch & tuning** — YIN, pYIN, piptrack, tuning estimation
//! - **Rhythm** — onset detection, beat tracking, tempo estimation, PLP
//! - **Effects** — HPSS, time stretch, pitch shift, trim/split, preemphasis
//! - **Conversions** — Hz/MIDI/note/mel/bark, time/sample/frame, A/B/C/D
//!   frequency weighting, Indian notation (svara), FJS intervals
//! - **Utilities** — DTW, NMF, Viterbi decoding, recurrence analysis,
//!   segmentation, peak picking
//!
//! # Quick Start
//!
//! ```rust
//! use giggle::{io, feature, spectrum};
//!
//! // Generate a 440 Hz tone (1 second at 22050 Hz)
//! let signal = io::tone(440.0, 22050, 1.0);
//!
//! // Compute mel spectrogram
//! let mel = feature::mel::melspectrogram(&signal, 22050, 2048, 512, 128).unwrap();
//! assert_eq!(mel.shape()[0], 128); // 128 mel bands
//!
//! // Compute MFCCs
//! let mfcc = feature::mfcc::mfcc(&signal, 22050, 13, 2048, 512, 128).unwrap();
//! assert_eq!(mfcc.shape()[0], 13); // 13 coefficients
//! ```
//!
//! # Modules
//!
//! | Module | Description |
//! |--------|-------------|
//! | [`io`] | Audio I/O, resampling, signal generators (`tone`, `clicks`, `chirp`) |
//! | [`spectrum`] | STFT/ISTFT, Griffin-Lim, dB conversions, PCEN, reassignment |
//! | [`cqt`] | Constant-Q and Variable-Q transforms |
//! | [`feature`] | Mel, MFCC, chroma, spectral features, tempo |
//! | [`pitch`] | YIN, pYIN, piptrack, tuning estimation |
//! | [`onset`] | Onset strength and detection |
//! | [`beat`] | Beat tracking and PLP |
//! | [`effects`] | HPSS, time stretch, pitch shift, trim/split |
//! | [`harmonic`] | Harmonic salience and interpolation |
//! | [`filters`] | Filterbanks (CQ, wavelet, semitone), IIR filtering |
//! | [`convert`] | Frequency/pitch/time conversions, music theory |
//! | [`utils`] | DTW, NMF, Viterbi, segmentation, peak picking |
//! | [`window`] | Window functions (Hann, Hamming, Blackman, etc.) |
//! | [`frame`] | Signal framing utilities |
//! | [`files`] | Audio file discovery and example data |
//!
//! # Error Handling
//!
//! All fallible operations return [`Result<T>`], which is an alias for
//! `std::result::Result<T, Error>`. The [`Error`] enum covers invalid
//! parameters, empty audio, shape mismatches, and I/O failures.
//!
//! # Safety
//!
//! This crate uses `#![forbid(unsafe_code)]` — no unsafe Rust anywhere.
//!
//! # Feature Flags
//!
//! | Flag | Description |
//! |------|-------------|
//! | `display` | PPM-based spectrogram and waveform visualization |

#![forbid(unsafe_code)]

pub mod error;
pub use error::{Error, Result};

pub mod beat;
pub mod convert;
pub mod cqt;
pub mod effects;
pub mod feature;
pub mod fft;
pub mod files;
pub mod filters;
pub mod frame;
pub mod harmonic;
pub mod io;
pub mod onset;
pub mod pitch;
pub mod spectrum;
pub mod utils;
pub mod window;

#[cfg(feature = "display")]
pub mod display;
