# Giggle 🎵

Audio and music analysis library for Rust.

Giggle is a Rust port of Python's [librosa](https://librosa.org/), covering the full pipeline from audio I/O through spectral analysis and feature extraction.

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
giggle = { git = "https://github.com/GreatV/giggle.git" }
```

## Quick Start

```rust
use giggle::{io, feature, spectrum};

// Generate a 440 Hz tone (1 second at 22050 Hz)
let signal = io::tone(440.0, 22050, 1.0);

// Compute mel spectrogram
let mel = feature::mel::melspectrogram(&signal, 22050, 2048, 512, 128).unwrap();
assert_eq!(mel.shape()[0], 128); // 128 mel bands

// Compute MFCCs
let mfcc = feature::mfcc::mfcc(&signal, 22050, 13, 2048, 512, 128).unwrap();
assert_eq!(mfcc.shape()[0], 13); // 13 coefficients
```

### Pitch Estimation

```rust
use giggle::pitch::YinConfig;

let signal = giggle::io::tone(440.0, 22050, 1.0);
let config = YinConfig::new(22050)
    .with_fmin(40.0)
    .with_fmax(5000.0);
let f0 = config.yin(&signal).unwrap();
```

### Effects

```rust
use giggle::{io, effects};

let signal = io::tone(440.0, 22050, 1.0);
// Harmonic-percussive source separation
let (harmonic, percussive) = effects::hpss::hpss(&signal, 2048, 512, 31).unwrap();
```

## Acknowledgments

Inspired by [librosa](https://librosa.org/) by Brian McFee et al.
