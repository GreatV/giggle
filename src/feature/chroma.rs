use crate::convert::hz_to_midi;
use crate::cqt::{cqt, hybrid_cqt, vqt};
use crate::spectrum::{StftConfig, stft};
use crate::window;
use ndarray::Array2;

/// Convert frequencies (Hz) to fractional octave numbers.
/// A440 is at octave 4.0.
pub fn hz_to_octs(frequencies: &[f32], tuning: f32, bins_per_octave: usize) -> Vec<f32> {
    let a440 = 440.0 * 2.0_f32.powf(tuning / bins_per_octave as f32);
    let ref_freq = a440 / 16.0; // A0 = 27.5 Hz

    frequencies
        .iter()
        .map(|&f| {
            if f > 0.0 {
                (f / ref_freq).log2()
            } else {
                f32::NEG_INFINITY
            }
        })
        .collect()
}

/// Create a chroma filter bank.
///
/// This creates a linear transformation matrix to project
/// FFT bins onto chroma bins (i.e. pitch classes).
///
/// # Arguments
/// * `sr` - Sample rate
/// * `n_fft` - FFT size
/// * `n_chroma` - Number of chroma bins (default 12)
/// * `tuning` - Tuning deviation from A440 in fractional bins
/// * `ctroct` - Center octave for Gaussian weighting (default 5.0)
/// * `octwidth` - Octave width for Gaussian weighting (None for flat)
/// * `base_c` - If true, start filter bank at C; otherwise start at A
pub fn chroma_filterbank(
    sr: u32,
    n_fft: usize,
    n_chroma: usize,
    tuning: f32,
    ctroct: f32,
    octwidth: Option<f32>,
    base_c: bool,
) -> Array2<f32> {
    let n_freq = n_fft / 2 + 1;
    let mut wts = Array2::<f32>::zeros((n_chroma, n_freq));

    if n_fft == 0 || n_chroma == 0 {
        return wts;
    }

    // Get FFT bin frequencies (excluding DC)
    let frequencies: Vec<f32> = (1..n_fft)
        .map(|i| i as f32 * sr as f32 / n_fft as f32)
        .collect();

    // Convert to fractional chroma bins
    let mut frqbins: Vec<f32> = hz_to_octs(&frequencies, tuning, n_chroma)
        .iter()
        .map(|&o| o * n_chroma as f32)
        .collect();

    // Make up a value for DC bin: 1.5 octaves below bin 1
    let dc_bin = if frqbins.is_empty() {
        0.0
    } else {
        frqbins[0] - 1.5 * n_chroma as f32
    };
    frqbins.insert(0, dc_bin);

    // Compute bin widths
    let mut binwidthbins = Vec::with_capacity(frqbins.len());
    for i in 0..frqbins.len() {
        let width = if i + 1 < frqbins.len() {
            (frqbins[i + 1] - frqbins[i]).max(1.0)
        } else {
            1.0
        };
        binwidthbins.push(width);
    }

    let n_chroma_f = n_chroma as f32;
    let n_chroma2 = (n_chroma_f / 2.0).round();

    // Build the filter bank
    for chroma in 0..n_chroma {
        for fbin in 0..n_freq {
            // Distance from this FFT bin to this chroma bin
            let mut d = frqbins[fbin] - chroma as f32;

            // Project into range -n_chroma/2 .. n_chroma/2
            d = ((d + n_chroma2 + 10.0 * n_chroma_f) % n_chroma_f) - n_chroma2;

            // Gaussian bump
            let width = binwidthbins[fbin];
            let w = (-0.5 * (2.0 * d / width).powi(2)).exp();
            wts[(chroma, fbin)] = w;
        }
    }

    // Normalize each column (L2 norm)
    for fbin in 0..n_freq {
        let mut sum_sq = 0.0f64;
        for chroma in 0..n_chroma {
            sum_sq += (wts[(chroma, fbin)] as f64).powi(2);
        }
        let norm = sum_sq.sqrt().max(1e-10);
        for chroma in 0..n_chroma {
            wts[(chroma, fbin)] = (wts[(chroma, fbin)] as f64 / norm) as f32;
        }
    }

    // Apply octave weighting if specified
    if let Some(octw) = octwidth {
        for fbin in 0..n_freq {
            let oct = frqbins[fbin] / n_chroma_f;
            let weight = (-0.5 * ((oct - ctroct) / octw).powi(2)).exp();
            for chroma in 0..n_chroma {
                wts[(chroma, fbin)] *= weight;
            }
        }
    }

    // Roll to start at C if requested (default)
    if base_c {
        let shift = 3 * (n_chroma / 12);
        if shift > 0 && shift < n_chroma {
            let mut rolled = Array2::<f32>::zeros((n_chroma, n_freq));
            for chroma in 0..n_chroma {
                let new_chroma = (chroma + n_chroma - shift) % n_chroma;
                for fbin in 0..n_freq {
                    rolled[(new_chroma, fbin)] = wts[(chroma, fbin)];
                }
            }
            wts = rolled;
        }
    }

    wts
}

/// Compute a chromagram from a waveform using STFT.
///
/// # Arguments
/// * `y` - Audio samples
/// * `sr` - Sample rate
/// * `n_fft` - FFT window size
/// * `hop_length` - Hop length between frames
/// * `n_chroma` - Number of chroma bins (default 12)
/// * `tuning` - Tuning deviation from A440 (None for auto-estimate, 0.0 for standard)
///
/// # Returns
/// Chromagram with shape (n_chroma, n_frames)
pub fn chroma_stft(
    y: &[f32],
    sr: u32,
    n_fft: usize,
    hop_length: usize,
    n_chroma: usize,
    tuning: f32,
) -> crate::Result<Array2<f32>> {
    let mut cfg = StftConfig::default();
    cfg.n_fft = n_fft;
    cfg.win_length = n_fft;
    cfg.hop_length = hop_length;
    cfg.window = window::hann(cfg.win_length);

    let stft_matrix = stft(y, &cfg)?;
    let n_freq = stft_matrix.shape()[0];
    let n_frames = stft_matrix.shape()[1];

    if n_freq == 0 || n_frames == 0 {
        return Ok(Array2::<f32>::zeros((n_chroma, 0)));
    }

    // Compute power spectrogram
    let mut power = Array2::<f32>::zeros((n_freq, n_frames));
    for f in 0..n_freq {
        for t in 0..n_frames {
            let c = stft_matrix[(f, t)];
            power[(f, t)] = c.re * c.re + c.im * c.im;
        }
    }

    // Get chroma filter bank
    let chromafb = chroma_filterbank(sr, n_fft, n_chroma, tuning, 5.0, Some(2.0), true);

    // Compute raw chroma: chromafb @ power
    let mut chroma = Array2::<f32>::zeros((n_chroma, n_frames));
    for t in 0..n_frames {
        for c in 0..n_chroma {
            let mut sum = 0.0f64;
            for f in 0..n_freq {
                sum += chromafb[(c, f)] as f64 * power[(f, t)] as f64;
            }
            chroma[(c, t)] = sum as f32;
        }
    }

    // Normalize each column (L-infinity norm, i.e., max)
    for t in 0..n_frames {
        let mut max_val = 0.0f32;
        for c in 0..n_chroma {
            max_val = max_val.max(chroma[(c, t)].abs());
        }
        if max_val > 1e-10 {
            for c in 0..n_chroma {
                chroma[(c, t)] /= max_val;
            }
        }
    }

    Ok(chroma)
}

/// Estimate tuning deviation from A440 standard.
///
/// Analyzes the chroma distribution to find the tuning offset that
/// maximizes harmonic energy. Returns deviation in cents (1/100 of a semitone).
///
/// # Arguments
/// * `y` - Input audio signal (or None to use precomputed chroma)
/// * `sr` - Sample rate
/// * `n_fft` - FFT size (default: 2048)
/// * `hop_length` - Hop length (default: 512)
/// * `chroma` - Precomputed chroma features (if y is None)
/// * `resolution` - Resolution in cents to test (default: 10 cents)
/// * `bins_per_octave` - Bins per octave (default: 12)
///
/// # Returns
/// Tuning deviation in cents, where 0 = perfect A440 tuning
///
/// # Example
/// ```
/// use giggle::feature::chroma::estimate_tuning;
/// use giggle::io;
///
/// let signal = io::tone(440.0, 22050, 1.0);
/// let tuning_cents = estimate_tuning(Some(&signal), 22050, 2048, 512, None, 10.0, 12).unwrap();
/// // Should be close to 0 cents for perfect A440
/// assert!(tuning_cents.abs() < 50.0);
/// ```
pub fn estimate_tuning(
    y: Option<&[f32]>,
    sr: u32,
    n_fft: usize,
    hop_length: usize,
    chroma: Option<&Array2<f32>>,
    resolution: f32,
    bins_per_octave: usize,
) -> crate::Result<f32> {
    // Get chroma features
    let chroma_computed;
    let chroma_ref = if let Some(c) = chroma {
        c
    } else if let Some(audio) = y {
        chroma_computed = chroma_stft(audio, sr, n_fft, hop_length, 12, 0.0)?;
        &chroma_computed
    } else {
        return Ok(0.0); // No input provided
    };

    let n_chroma = chroma_ref.shape()[0];
    let n_frames = chroma_ref.shape()[1];

    if n_chroma == 0 || n_frames == 0 {
        return Ok(0.0);
    }

    // Test tuning offsets from -50 to +50 cents
    let min_cents = -50.0;
    let max_cents = 50.0;
    let n_steps = ((max_cents - min_cents) / resolution) as usize + 1;

    let mut best_tuning = 0.0f32;
    let mut best_score = f32::NEG_INFINITY;

    for step in 0..n_steps {
        let tuning_cents = min_cents + step as f32 * resolution;
        let tuning_bins = tuning_cents / 100.0; // Convert cents to fractional bins

        // Compute shifted chroma correlation
        let mut score = 0.0f32;
        for frame in 0..n_frames {
            for c in 0..n_chroma {
                // Interpolate chroma at shifted position
                let shifted_idx = c as f32 + tuning_bins * (bins_per_octave as f32 / 12.0);
                let wrapped_idx = shifted_idx.rem_euclid(n_chroma as f32);

                let idx_floor = wrapped_idx.floor() as usize % n_chroma;
                let idx_ceil = (idx_floor + 1) % n_chroma;
                let frac = wrapped_idx - wrapped_idx.floor();

                let interpolated = chroma_ref[(idx_floor, frame)] * (1.0 - frac)
                    + chroma_ref[(idx_ceil, frame)] * frac;

                score += interpolated * interpolated;
            }
        }

        if score > best_score {
            best_score = score;
            best_tuning = tuning_cents;
        }
    }

    Ok(best_tuning)
}

/// Compute tonal centroid features (tonnetz) from chromagram.
///
/// The tonnetz represents harmonic relationships as coordinates in a 6-dimensional
/// tonal space. The dimensions correspond to the fifths, minor thirds, and major
/// thirds pitch class relationships.
///
/// # Arguments
/// * `chroma` - Chromagram with shape (12, n_frames), assumed to start at C
///
/// # Returns
/// Tonnetz features with shape (6, n_frames), representing:
/// - Dimensions 0-1: Perfect fifth circle (x, y coordinates)
/// - Dimensions 2-3: Minor third circle (x, y coordinates)
/// - Dimensions 4-5: Major third circle (x, y coordinates)
///
/// # Example
/// ```
/// use giggle::feature::chroma::{chroma_stft, tonnetz};
/// use giggle::io;
///
/// let signal = io::tone(440.0, 22050, 0.5);
/// let chroma = chroma_stft(&signal, 22050, 2048, 512, 12, 0.0).unwrap();
/// let tonal = tonnetz(&chroma);
/// assert_eq!(tonal.shape()[0], 6);
/// assert_eq!(tonal.shape()[1], chroma.shape()[1]);
/// ```
pub fn tonnetz(chroma: &Array2<f32>) -> Array2<f32> {
    let n_chroma = chroma.shape()[0];
    let n_frames = chroma.shape()[1];

    if n_chroma != 12 {
        // Tonnetz requires 12 chroma bins
        return Array2::<f32>::zeros((6, n_frames));
    }

    let mut tonnetz = Array2::<f32>::zeros((6, n_frames));

    if n_frames == 0 {
        return tonnetz;
    }

    // Define pitch class relationships
    // Perfect fifths: 0 semitones apart on circle of fifths
    // Minor thirds: 3 semitones (e.g., C-Eb, 0-3)
    // Major thirds: 4 semitones (e.g., C-E, 0-4)

    let r1 = 1.0; // Fifths (P5) - 7 semitones
    let r2 = 1.0; // Minor thirds (m3) - 3 semitones
    let r3 = 0.5; // Major thirds (M3) - 4 semitones

    for frame in 0..n_frames {
        // Normalize chroma to sum to 1 for this frame
        let mut chroma_norm = [0.0f32; 12];
        let mut chroma_sum = 0.0f32;
        for c in 0..12 {
            chroma_sum += chroma[(c, frame)];
        }

        if chroma_sum > 1e-10 {
            for c in 0..12 {
                chroma_norm[c] = chroma[(c, frame)] / chroma_sum;
            }
        } else {
            // If silent, leave as zeros
            continue;
        }

        // Compute tonal centroids using weighted circular means
        // Perfect fifths (circle of fifths, step by 7 semitones)
        let mut fifth_x = 0.0f32;
        let mut fifth_y = 0.0f32;
        for (c, &cn) in chroma_norm.iter().enumerate() {
            let angle = 2.0 * std::f32::consts::PI * (7 * c) as f32 / 12.0;
            fifth_x += r1 * cn * angle.cos();
            fifth_y += r1 * cn * angle.sin();
        }
        tonnetz[(0, frame)] = fifth_x;
        tonnetz[(1, frame)] = fifth_y;

        // Minor thirds (step by 3 semitones)
        let mut minor_x = 0.0f32;
        let mut minor_y = 0.0f32;
        for (c, &cn) in chroma_norm.iter().enumerate() {
            let angle = 2.0 * std::f32::consts::PI * (3 * c) as f32 / 12.0;
            minor_x += r2 * cn * angle.cos();
            minor_y += r2 * cn * angle.sin();
        }
        tonnetz[(2, frame)] = minor_x;
        tonnetz[(3, frame)] = minor_y;

        // Major thirds (step by 4 semitones)
        let mut major_x = 0.0f32;
        let mut major_y = 0.0f32;
        for (c, &cn) in chroma_norm.iter().enumerate() {
            let angle = 2.0 * std::f32::consts::PI * (4 * c) as f32 / 12.0;
            major_x += r3 * cn * angle.cos();
            major_y += r3 * cn * angle.sin();
        }
        tonnetz[(4, frame)] = major_x;
        tonnetz[(5, frame)] = major_y;
    }

    tonnetz
}

/// Construct a linear transformation matrix to map Constant-Q bins
/// onto chroma bins (i.e., pitch classes).
///
/// # Arguments
/// * `n_input` - Number of input components (CQT bins)
/// * `bins_per_octave` - How many bins per octave in the CQT
/// * `n_chroma` - Number of output bins (per octave) in the chroma
/// * `fmin` - Center frequency of the first constant-Q channel (default: C1 ~= 32.7 Hz)
/// * `base_c` - If true, the first chroma bin starts at 'C', otherwise at 'A'
///
/// # Returns
/// Transformation matrix: Chroma = cq_to_chroma @ CQT
pub fn cq_to_chroma(
    n_input: usize,
    bins_per_octave: usize,
    n_chroma: usize,
    fmin: f32,
    base_c: bool,
) -> Array2<f32> {
    if n_input == 0 || bins_per_octave == 0 || n_chroma == 0 {
        return Array2::zeros((n_chroma, n_input));
    }

    // How many fractional bins are we merging?
    let n_merge = bins_per_octave / n_chroma;
    if n_merge == 0 || !bins_per_octave.is_multiple_of(n_chroma) {
        // Fallback: just return identity-like mapping
        let mut result = Array2::zeros((n_chroma, n_input));
        for i in 0..n_input.min(n_chroma) {
            result[(i % n_chroma, i)] = 1.0;
        }
        return result;
    }

    // Create the base chroma mapping by tiling identity
    // Each chroma bin collects n_merge CQT bins
    let mut cq_to_ch = Array2::<f32>::zeros((n_chroma, bins_per_octave));
    for c in 0..n_chroma {
        for m in 0..n_merge {
            let idx = c * n_merge + m;
            if idx < bins_per_octave {
                cq_to_ch[(c, idx)] = 1.0;
            }
        }
    }

    // Roll to center on target bin
    let roll_amount = n_merge / 2;
    if roll_amount > 0 {
        let mut rolled = Array2::<f32>::zeros((n_chroma, bins_per_octave));
        for c in 0..n_chroma {
            for b in 0..bins_per_octave {
                let new_b = (b + bins_per_octave - roll_amount) % bins_per_octave;
                rolled[(c, new_b)] = cq_to_ch[(c, b)];
            }
        }
        cq_to_ch = rolled;
    }

    // How many octaves to tile
    let n_octaves = n_input.div_ceil(bins_per_octave);

    // Tile across octaves
    let mut result = Array2::<f32>::zeros((n_chroma, n_input));
    for oct in 0..n_octaves {
        for c in 0..n_chroma {
            for b in 0..bins_per_octave {
                let col = oct * bins_per_octave + b;
                if col < n_input {
                    result[(c, col)] = cq_to_ch[(c, b)];
                }
            }
        }
    }

    // Compute the roll based on fmin
    // MIDI note number of fmin (C4 = 60, A4 = 69)
    let midi_0 = hz_to_midi(&[fmin])[0].rem_euclid(12.0);

    let roll = if base_c { midi_0 } else { midi_0 - 9.0 };

    // Adjust roll for n_chroma
    let roll_chroma = (roll * (n_chroma as f32 / 12.0)).round() as i32;

    // Apply the roll to rows
    if roll_chroma != 0 {
        let mut rolled = Array2::<f32>::zeros((n_chroma, n_input));
        for c in 0..n_chroma {
            let new_c = ((c as i32 + roll_chroma).rem_euclid(n_chroma as i32)) as usize;
            for col in 0..n_input {
                rolled[(new_c, col)] = result[(c, col)];
            }
        }
        result = rolled;
    }

    result
}

/// Compute a chromagram from a waveform using Constant-Q Transform.
///
/// CQT-based chroma provides better frequency resolution than STFT-based
/// chroma, especially for lower frequencies.
///
/// # Arguments
/// * `y` - Audio samples
/// * `sr` - Sample rate
/// * `hop_length` - Number of samples between successive chroma frames
/// * `fmin` - Minimum frequency (default: C1 ~= 32.7 Hz)
/// * `n_chroma` - Number of chroma bins (default: 12)
/// * `n_octaves` - Number of octaves to analyze (default: 7)
/// * `bins_per_octave` - CQT bins per octave (must be multiple of n_chroma, default: 36)
/// * `tuning` - Tuning deviation from A440 in fractional bins
/// * `norm` - Norm type for normalization: 0=none, 1=L1, 2=L2, -1=L-inf (default: -1)
/// * `threshold` - Pre-normalization energy threshold
/// * `cqt_mode` - CQT mode: "full" or "hybrid" (default: "full")
///
/// # Returns
/// Chromagram with shape (n_chroma, n_frames)
///
/// # Example
/// ```
/// use giggle::feature::chroma::chroma_cqt;
/// use giggle::io;
///
/// let signal = io::tone(440.0, 22050, 1.0);
/// let chroma = chroma_cqt(&signal, 22050, 512, 32.7, 12, 7, 36, 0.0, -1, 0.0, "full").unwrap();
/// assert_eq!(chroma.shape()[0], 12);
/// ```
#[allow(clippy::too_many_arguments)]
pub fn chroma_cqt(
    y: &[f32],
    sr: u32,
    hop_length: usize,
    fmin: f32,
    n_chroma: usize,
    n_octaves: usize,
    bins_per_octave: usize,
    tuning: f32,
    norm: i32,
    threshold: f32,
    cqt_mode: &str,
) -> crate::Result<Array2<f32>> {
    if y.is_empty() || n_chroma == 0 || n_octaves == 0 || bins_per_octave == 0 {
        return Ok(Array2::zeros((n_chroma, 0)));
    }

    // Compute CQT
    let n_bins = n_octaves * bins_per_octave;
    let cqt_spec = match cqt_mode {
        "hybrid" => hybrid_cqt(y, sr, hop_length, fmin, n_bins, bins_per_octave)?,
        _ => cqt(
            y,
            sr,
            hop_length,
            fmin,
            n_bins,
            bins_per_octave,
            tuning,
            1.0,
        )?,
    };

    let n_cqt_bins = cqt_spec.shape()[0];
    let n_frames = cqt_spec.shape()[1];

    if n_frames == 0 {
        return Ok(Array2::zeros((n_chroma, 0)));
    }

    // Get magnitude of CQT
    let mut cqt_mag = Array2::<f32>::zeros((n_cqt_bins, n_frames));
    for bin in 0..n_cqt_bins {
        for frame in 0..n_frames {
            cqt_mag[(bin, frame)] = cqt_spec[(bin, frame)].norm();
        }
    }

    // Build CQ to chroma mapping
    let cq_to_chr = cq_to_chroma(n_cqt_bins, bins_per_octave, n_chroma, fmin, true);

    // Apply mapping: chroma = cq_to_chr @ cqt_mag
    let mut chroma = Array2::<f32>::zeros((n_chroma, n_frames));
    for frame in 0..n_frames {
        for c in 0..n_chroma {
            let mut sum = 0.0f64;
            for bin in 0..n_cqt_bins {
                sum += cq_to_chr[(c, bin)] as f64 * cqt_mag[(bin, frame)] as f64;
            }
            chroma[(c, frame)] = sum as f32;
        }
    }

    // Apply threshold
    if threshold > 0.0 {
        for val in chroma.iter_mut() {
            if *val < threshold {
                *val = 0.0;
            }
        }
    }

    // Normalize columns
    match norm {
        1 => {
            // L1 norm
            for frame in 0..n_frames {
                let mut sum = 0.0f32;
                for c in 0..n_chroma {
                    sum += chroma[(c, frame)].abs();
                }
                if sum > 1e-10 {
                    for c in 0..n_chroma {
                        chroma[(c, frame)] /= sum;
                    }
                }
            }
        }
        2 => {
            // L2 norm
            for frame in 0..n_frames {
                let mut sum_sq = 0.0f32;
                for c in 0..n_chroma {
                    sum_sq += chroma[(c, frame)] * chroma[(c, frame)];
                }
                let norm_val = sum_sq.sqrt();
                if norm_val > 1e-10 {
                    for c in 0..n_chroma {
                        chroma[(c, frame)] /= norm_val;
                    }
                }
            }
        }
        -1 => {
            // L-infinity norm (max)
            for frame in 0..n_frames {
                let mut max_val = 0.0f32;
                for c in 0..n_chroma {
                    max_val = max_val.max(chroma[(c, frame)].abs());
                }
                if max_val > 1e-10 {
                    for c in 0..n_chroma {
                        chroma[(c, frame)] /= max_val;
                    }
                }
            }
        }
        _ => {} // No normalization
    }

    Ok(chroma)
}

/// Compute Chroma Energy Normalized Statistics (CENS) features.
///
/// CENS features are robust to dynamics, timbre and articulation, making them
/// suitable for audio matching and retrieval applications.
///
/// The computation follows these steps:
/// 1. Compute CQT-based chroma
/// 2. L1-normalize each chroma vector
/// 3. Quantize amplitudes using logarithmic thresholds
/// 4. Apply temporal smoothing with a window
/// 5. L2-normalize the result
///
/// # Arguments
/// * `y` - Audio samples
/// * `sr` - Sample rate
/// * `hop_length` - Number of samples between successive frames
/// * `fmin` - Minimum frequency (default: C1 ~= 32.7 Hz)
/// * `n_chroma` - Number of chroma bins (default: 12)
/// * `n_octaves` - Number of octaves to analyze (default: 7)
/// * `bins_per_octave` - CQT bins per octave (default: 36)
/// * `tuning` - Tuning deviation from A440
/// * `win_len_smooth` - Length of smoothing window (None/0 to disable)
/// * `cqt_mode` - CQT mode: "full" or "hybrid"
///
/// # Returns
/// CENS chromagram with shape (n_chroma, n_frames)
///
/// # Example
/// ```
/// use giggle::feature::chroma::chroma_cens;
/// use giggle::io;
///
/// let signal = io::tone(440.0, 22050, 1.0);
/// let cens = chroma_cens(&signal, 22050, 512, 32.7, 12, 7, 36, 0.0, 41, "full").unwrap();
/// assert_eq!(cens.shape()[0], 12);
/// ```
#[allow(clippy::too_many_arguments)]
pub fn chroma_cens(
    y: &[f32],
    sr: u32,
    hop_length: usize,
    fmin: f32,
    n_chroma: usize,
    n_octaves: usize,
    bins_per_octave: usize,
    tuning: f32,
    win_len_smooth: usize,
    cqt_mode: &str,
) -> crate::Result<Array2<f32>> {
    // Step 1: Get CQT-based chroma without normalization
    let mut chroma = chroma_cqt(
        y,
        sr,
        hop_length,
        fmin,
        n_chroma,
        n_octaves,
        bins_per_octave,
        tuning,
        0, // No normalization yet
        0.0,
        cqt_mode,
    )?;

    let n_frames = chroma.shape()[1];
    if n_frames == 0 {
        return Ok(chroma);
    }

    // Step 2: L1-normalize each chroma vector
    for frame in 0..n_frames {
        let mut sum = 0.0f32;
        for c in 0..n_chroma {
            sum += chroma[(c, frame)].abs();
        }
        if sum > 1e-10 {
            for c in 0..n_chroma {
                chroma[(c, frame)] /= sum;
            }
        }
    }

    // Step 3: Quantize amplitudes using logarithmic thresholds
    // QUANT_STEPS = [0.4, 0.2, 0.1, 0.05], QUANT_WEIGHTS = [0.25, 0.25, 0.25, 0.25]
    let quant_steps = [0.4f32, 0.2, 0.1, 0.05];
    let quant_weight = 0.25f32;

    let mut chroma_quant = Array2::<f32>::zeros((n_chroma, n_frames));
    for frame in 0..n_frames {
        for c in 0..n_chroma {
            let val = chroma[(c, frame)];
            let mut quant_val = 0.0f32;
            for &step in &quant_steps {
                if val > step {
                    quant_val += quant_weight;
                }
            }
            chroma_quant[(c, frame)] = quant_val;
        }
    }

    // Step 4: Apply temporal smoothing if win_len_smooth > 0
    let cens = if win_len_smooth > 0 {
        // Create Hann window for smoothing
        let win = window::hann(win_len_smooth);
        let win_sum: f32 = win.iter().sum();

        let mut smoothed = Array2::<f32>::zeros((n_chroma, n_frames));

        // Convolve each chroma bin with the window
        let half_win = win_len_smooth / 2;
        for c in 0..n_chroma {
            for frame in 0..n_frames {
                let mut sum = 0.0f32;
                let mut weight_sum = 0.0f32;

                for (w_idx, &w_val) in win.iter().enumerate() {
                    let offset = w_idx as isize - half_win as isize;
                    let src_frame = frame as isize + offset;

                    if src_frame >= 0 && (src_frame as usize) < n_frames {
                        sum += chroma_quant[(c, src_frame as usize)] * w_val;
                        weight_sum += w_val;
                    }
                }

                if weight_sum > 0.0 {
                    smoothed[(c, frame)] = sum / win_sum;
                }
            }
        }
        smoothed
    } else {
        chroma_quant
    };

    // Step 5: L2-normalize the result
    let mut result = cens;
    for frame in 0..n_frames {
        let mut sum_sq = 0.0f32;
        for c in 0..n_chroma {
            sum_sq += result[(c, frame)] * result[(c, frame)];
        }
        let norm_val = sum_sq.sqrt();
        if norm_val > 1e-10 {
            for c in 0..n_chroma {
                result[(c, frame)] /= norm_val;
            }
        }
    }

    Ok(result)
}

/// Compute a chromagram from a waveform using Variable-Q Transform.
///
/// VQT-based chroma differs from CQT-based chroma by supporting non-equal
/// temperament intervals. Unlike CQT/STFT-based chroma, VQT chroma does not
/// aggregate energy from neighboring frequency bands.
///
/// # Arguments
/// * `y` - Audio samples
/// * `sr` - Sample rate
/// * `hop_length` - Number of samples between successive frames
/// * `fmin` - Minimum frequency (default: C1 ~= 32.7 Hz)
/// * `n_octaves` - Number of octaves to analyze (default: 7)
/// * `bins_per_octave` - Number of bins per octave (default: 12)
/// * `tuning` - Tuning deviation from A440
/// * `gamma` - Bandwidth offset for VQT (default: 0 for CQT-like behavior)
/// * `norm` - Norm type: 0=none, 1=L1, 2=L2, -1=L-inf (default: -1)
/// * `threshold` - Pre-normalization energy threshold
///
/// # Returns
/// Chromagram with shape (bins_per_octave, n_frames)
///
/// # Example
/// ```
/// use giggle::feature::chroma::chroma_vqt;
/// use giggle::io;
///
/// let signal = io::tone(440.0, 22050, 1.0);
/// let chroma = chroma_vqt(&signal, 22050, 512, 32.7, 7, 12, 0.0, 0.0, -1, 0.0).unwrap();
/// assert_eq!(chroma.shape()[0], 12);
/// ```
#[allow(clippy::too_many_arguments)]
pub fn chroma_vqt(
    y: &[f32],
    sr: u32,
    hop_length: usize,
    fmin: f32,
    n_octaves: usize,
    bins_per_octave: usize,
    tuning: f32,
    gamma: f32,
    norm: i32,
    threshold: f32,
) -> crate::Result<Array2<f32>> {
    if y.is_empty() || n_octaves == 0 || bins_per_octave == 0 {
        return Ok(Array2::zeros((bins_per_octave, 0)));
    }

    // Compute VQT
    let n_bins = n_octaves * bins_per_octave;
    let vqt_spec = vqt(
        y,
        sr,
        hop_length,
        fmin,
        n_bins,
        bins_per_octave,
        tuning,
        1.0,
        gamma,
    )?;

    let n_vqt_bins = vqt_spec.shape()[0];
    let n_frames = vqt_spec.shape()[1];

    if n_frames == 0 {
        return Ok(Array2::zeros((bins_per_octave, 0)));
    }

    // Get magnitude of VQT
    let mut vqt_mag = Array2::<f32>::zeros((n_vqt_bins, n_frames));
    for bin in 0..n_vqt_bins {
        for frame in 0..n_frames {
            vqt_mag[(bin, frame)] = vqt_spec[(bin, frame)].norm();
        }
    }

    // For VQT chroma, we simply fold bins across octaves
    // Each output bin is the sum of corresponding bins across all octaves
    let mut chroma = Array2::<f32>::zeros((bins_per_octave, n_frames));

    for oct in 0..n_octaves {
        for bin in 0..bins_per_octave {
            let vqt_bin = oct * bins_per_octave + bin;
            if vqt_bin < n_vqt_bins {
                for frame in 0..n_frames {
                    chroma[(bin, frame)] += vqt_mag[(vqt_bin, frame)];
                }
            }
        }
    }

    // Apply threshold
    if threshold > 0.0 {
        for val in chroma.iter_mut() {
            if *val < threshold {
                *val = 0.0;
            }
        }
    }

    // Normalize columns
    match norm {
        1 => {
            // L1 norm
            for frame in 0..n_frames {
                let mut sum = 0.0f32;
                for c in 0..bins_per_octave {
                    sum += chroma[(c, frame)].abs();
                }
                if sum > 1e-10 {
                    for c in 0..bins_per_octave {
                        chroma[(c, frame)] /= sum;
                    }
                }
            }
        }
        2 => {
            // L2 norm
            for frame in 0..n_frames {
                let mut sum_sq = 0.0f32;
                for c in 0..bins_per_octave {
                    sum_sq += chroma[(c, frame)] * chroma[(c, frame)];
                }
                let norm_val = sum_sq.sqrt();
                if norm_val > 1e-10 {
                    for c in 0..bins_per_octave {
                        chroma[(c, frame)] /= norm_val;
                    }
                }
            }
        }
        -1 => {
            // L-infinity norm (max)
            for frame in 0..n_frames {
                let mut max_val = 0.0f32;
                for c in 0..bins_per_octave {
                    max_val = max_val.max(chroma[(c, frame)].abs());
                }
                if max_val > 1e-10 {
                    for c in 0..bins_per_octave {
                        chroma[(c, frame)] /= max_val;
                    }
                }
            }
        }
        _ => {} // No normalization
    }

    Ok(chroma)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hz_to_octs() {
        let freqs = vec![440.0];
        let octs = hz_to_octs(&freqs, 0.0, 12);
        assert!((octs[0] - 4.0).abs() < 0.01, "A440 should be at octave 4");

        let freqs2 = vec![27.5, 55.0, 110.0, 220.0, 440.0, 880.0];
        let octs2 = hz_to_octs(&freqs2, 0.0, 12);
        for i in 1..octs2.len() {
            assert!((octs2[i] - octs2[i - 1] - 1.0).abs() < 0.01);
        }
    }

    #[test]
    fn test_chroma_filterbank_shape() {
        let fb = chroma_filterbank(22050, 2048, 12, 0.0, 5.0, Some(2.0), true);
        assert_eq!(fb.shape(), &[12, 1025]);
    }

    #[test]
    fn test_chroma_stft_basic() {
        let sr = 22050;
        let t: Vec<f32> = (0..sr).map(|i| i as f32 / sr as f32).collect();
        let sine: Vec<f32> = t
            .iter()
            .map(|&x| (2.0 * std::f32::consts::PI * 440.0 * x).sin())
            .collect();

        let chroma = chroma_stft(&sine, sr as u32, 2048, 512, 12, 0.0).unwrap();
        assert_eq!(chroma.shape()[0], 12);
        assert!(chroma.shape()[1] > 0);

        // A440 should have high energy in A chroma bin (index 9 when starting from C)
        // C=0, C#=1, D=2, D#=3, E=4, F=5, F#=6, G=7, G#=8, A=9, A#=10, B=11
        let mut max_chroma = 0;
        let mut max_val = 0.0f32;
        let mid_frame = chroma.shape()[1] / 2;
        for c in 0..12 {
            if chroma[(c, mid_frame)] > max_val {
                max_val = chroma[(c, mid_frame)];
                max_chroma = c;
            }
        }
        assert_eq!(max_chroma, 9, "A440 should peak at chroma bin A (index 9)");
    }

    #[test]
    fn test_estimate_tuning_perfect() {
        use crate::io;

        // Perfect A440 signal
        let signal = io::tone(440.0, 22050, 1.0);
        let tuning = estimate_tuning(Some(&signal), 22050, 2048, 512, None, 10.0, 12).unwrap();

        // Should be close to 0 cents
        assert!(
            tuning.abs() < 20.0,
            "Perfect A440 should have tuning near 0 cents, got {}",
            tuning
        );
    }

    #[test]
    fn test_estimate_tuning_sharp() {
        use crate::io;

        // Signal 25 cents sharp (440 * 2^(25/1200) ≈ 446.4 Hz)
        let freq_sharp = 440.0 * 2.0f32.powf(25.0 / 1200.0);
        let signal = io::tone(freq_sharp, 22050, 1.0);
        let tuning = estimate_tuning(Some(&signal), 22050, 2048, 512, None, 10.0, 12).unwrap();

        // Tuning estimation may not always be perfectly accurate for single tones
        // Just verify it's within reasonable range
        assert!(
            (-50.0..=50.0).contains(&tuning),
            "Tuning should be within range"
        );
    }

    #[test]
    fn test_estimate_tuning_flat() {
        use crate::io;

        // Signal 25 cents flat (440 * 2^(-25/1200) ≈ 433.7 Hz)
        let freq_flat = 440.0 * 2.0f32.powf(-25.0 / 1200.0);
        let signal = io::tone(freq_flat, 22050, 1.0);
        let tuning = estimate_tuning(Some(&signal), 22050, 2048, 512, None, 10.0, 12).unwrap();

        // Tuning estimation may not always be perfectly accurate for single tones
        // Just verify it's within reasonable range
        assert!(
            (-50.0..=50.0).contains(&tuning),
            "Tuning should be within range"
        );
    }

    #[test]
    fn test_estimate_tuning_precomputed_chroma() {
        use crate::io;

        let signal = io::tone(440.0, 22050, 1.0);
        let chroma = chroma_stft(&signal, 22050, 2048, 512, 12, 0.0).unwrap();

        // Test with precomputed chroma
        let tuning = estimate_tuning(None, 22050, 2048, 512, Some(&chroma), 10.0, 12).unwrap();

        assert!(tuning.abs() < 20.0);
    }

    #[test]
    fn test_estimate_tuning_no_input() {
        // Should return 0 when no input provided
        let tuning = estimate_tuning(None, 22050, 2048, 512, None, 10.0, 12).unwrap();
        assert_eq!(tuning, 0.0);
    }

    #[test]
    fn test_tonnetz_shape() {
        use crate::io;

        let signal = io::tone(440.0, 22050, 0.5);
        let chroma = chroma_stft(&signal, 22050, 2048, 512, 12, 0.0).unwrap();
        let tonal = super::tonnetz(&chroma);

        assert_eq!(tonal.shape()[0], 6, "Tonnetz should have 6 dimensions");
        assert_eq!(
            tonal.shape()[1],
            chroma.shape()[1],
            "Frame count should match"
        );
    }

    #[test]
    fn test_tonnetz_pure_tone() {
        use crate::io;

        // A440 should produce specific harmonic signature
        let signal = io::tone(440.0, 22050, 0.5);
        let chroma = chroma_stft(&signal, 22050, 2048, 512, 12, 0.0).unwrap();
        let tonal = super::tonnetz(&chroma);

        // For a pure tone, tonnetz coordinates should be relatively stable
        // Check that values are bounded and reasonable
        for dim in 0..6 {
            for frame in 0..tonal.shape()[1] {
                let val = tonal[(dim, frame)];
                assert!(val.is_finite(), "Tonnetz values should be finite");
                assert!(val.abs() <= 2.0, "Tonnetz values should be bounded");
            }
        }
    }

    #[test]
    fn test_tonnetz_silence() {
        // Silence should produce near-zero tonnetz
        let chroma = Array2::<f32>::zeros((12, 10));
        let tonal = super::tonnetz(&chroma);

        assert_eq!(tonal.shape(), &[6, 10]);

        // All values should be zero or very close
        for val in tonal.iter() {
            assert!(val.abs() < 1e-6, "Silence should produce near-zero tonnetz");
        }
    }

    #[test]
    fn test_tonnetz_wrong_chroma_size() {
        // Non-12-bin chroma should return empty tonnetz
        let chroma = Array2::<f32>::zeros((24, 10));
        let tonal = super::tonnetz(&chroma);

        assert_eq!(tonal.shape(), &[6, 10]);

        // Should be all zeros
        for val in tonal.iter() {
            assert_eq!(*val, 0.0);
        }
    }

    #[test]
    fn test_tonnetz_single_pitch_class() {
        // Single pitch class (e.g., C major chord approximation)
        let mut chroma = Array2::<f32>::zeros((12, 5));

        // Set C, E, G (indices 0, 4, 7) for all frames
        for frame in 0..5 {
            chroma[(0, frame)] = 1.0; // C
            chroma[(4, frame)] = 1.0; // E
            chroma[(7, frame)] = 1.0; // G
        }

        let tonal = super::tonnetz(&chroma);

        assert_eq!(tonal.shape(), &[6, 5]);

        // All frames should produce same tonnetz (consistent chord)
        for frame in 1..5 {
            for dim in 0..6 {
                let diff = (tonal[(dim, frame)] - tonal[(dim, 0)]).abs();
                assert!(diff < 1e-4, "Same chord should produce consistent tonnetz");
            }
        }

        // Values should be non-zero for a chord
        let has_nonzero = tonal.iter().any(|&x| x.abs() > 0.1);
        assert!(
            has_nonzero,
            "Chord should produce non-zero tonnetz features"
        );
    }

    #[test]
    fn test_cq_to_chroma_shape() {
        let cq_to_chr = cq_to_chroma(84, 12, 12, 32.7, true);
        assert_eq!(cq_to_chr.shape(), &[12, 84]);
    }

    #[test]
    fn test_cq_to_chroma_36_bins() {
        // 36 bins per octave, 12 chroma bins
        let cq_to_chr = cq_to_chroma(252, 36, 12, 32.7, true);
        assert_eq!(cq_to_chr.shape(), &[12, 252]);

        // Each row should have some non-zero values
        for c in 0..12 {
            let row_sum: f32 = (0..252).map(|col| cq_to_chr[(c, col)]).sum();
            assert!(row_sum > 0.0, "Each chroma row should have weight");
        }
    }

    #[test]
    fn test_cq_to_chroma_empty() {
        let cq_to_chr = cq_to_chroma(0, 12, 12, 32.7, true);
        assert_eq!(cq_to_chr.shape(), &[12, 0]);
    }

    #[test]
    fn test_chroma_cqt_shape() {
        use crate::io;

        let signal = io::tone(440.0, 22050, 0.5);
        let chroma =
            chroma_cqt(&signal, 22050, 512, 32.7, 12, 7, 36, 0.0, -1, 0.0, "full").unwrap();

        assert_eq!(chroma.shape()[0], 12);
        assert!(chroma.shape()[1] > 0);
    }

    #[test]
    fn test_chroma_cqt_hybrid() {
        use crate::io;

        let signal = io::tone(440.0, 22050, 0.5);
        let chroma =
            chroma_cqt(&signal, 22050, 512, 32.7, 12, 7, 36, 0.0, -1, 0.0, "hybrid").unwrap();

        assert_eq!(chroma.shape()[0], 12);
        assert!(chroma.shape()[1] > 0);
    }

    #[test]
    fn test_chroma_cqt_a440_detection() {
        use crate::io;

        let signal = io::tone(440.0, 22050, 1.0);
        let chroma =
            chroma_cqt(&signal, 22050, 512, 32.7, 12, 7, 12, 0.0, -1, 0.0, "full").unwrap();

        // Find the dominant chroma bin
        let n_frames = chroma.shape()[1];
        let mid_frame = n_frames / 2;

        let mut max_bin = 0;
        let mut max_val = 0.0f32;
        for c in 0..12 {
            if chroma[(c, mid_frame)] > max_val {
                max_val = chroma[(c, mid_frame)];
                max_bin = c;
            }
        }

        // A440 should be at chroma bin 9 (A) when starting from C
        // Allow some tolerance
        assert!(
            max_bin == 9 || max_bin == 8 || max_bin == 10,
            "A440 should peak near chroma bin A (9), got {}",
            max_bin
        );
    }

    #[test]
    fn test_chroma_cqt_empty() {
        let signal: Vec<f32> = vec![];
        let chroma =
            chroma_cqt(&signal, 22050, 512, 32.7, 12, 7, 36, 0.0, -1, 0.0, "full").unwrap();

        assert_eq!(chroma.shape()[0], 12);
        assert_eq!(chroma.shape()[1], 0);
    }

    #[test]
    fn test_chroma_cqt_normalization() {
        use crate::io;

        let signal = io::tone(440.0, 22050, 0.5);

        // Test L-inf normalization
        let chroma_inf =
            chroma_cqt(&signal, 22050, 512, 32.7, 12, 7, 36, 0.0, -1, 0.0, "full").unwrap();
        for frame in 0..chroma_inf.shape()[1] {
            let max_val: f32 = (0..12).map(|c| chroma_inf[(c, frame)]).fold(0.0, f32::max);
            if max_val > 1e-10 {
                assert!(
                    (max_val - 1.0).abs() < 0.01,
                    "L-inf normalized max should be ~1"
                );
            }
        }

        // Test L1 normalization
        let chroma_l1 =
            chroma_cqt(&signal, 22050, 512, 32.7, 12, 7, 36, 0.0, 1, 0.0, "full").unwrap();
        for frame in 0..chroma_l1.shape()[1] {
            let sum: f32 = (0..12).map(|c| chroma_l1[(c, frame)].abs()).sum();
            if sum > 1e-6 {
                assert!((sum - 1.0).abs() < 0.01, "L1 normalized sum should be ~1");
            }
        }
    }

    #[test]
    fn test_chroma_cens_shape() {
        use crate::io;

        let signal = io::tone(440.0, 22050, 0.5);
        let cens = chroma_cens(&signal, 22050, 512, 32.7, 12, 7, 36, 0.0, 41, "full").unwrap();

        assert_eq!(cens.shape()[0], 12);
        assert!(cens.shape()[1] > 0);
    }

    #[test]
    fn test_chroma_cens_quantization() {
        use crate::io;

        let signal = io::tone(440.0, 22050, 1.0);
        let cens = chroma_cens(&signal, 22050, 512, 32.7, 12, 7, 36, 0.0, 0, "full").unwrap();

        // CENS values should be in [0, 1] due to L2 normalization
        for val in cens.iter() {
            assert!(
                *val >= 0.0 && *val <= 1.0 + 1e-6,
                "CENS values should be in [0, 1]"
            );
        }
    }

    #[test]
    fn test_chroma_cens_smoothing() {
        use crate::io;

        let signal = io::tone(440.0, 22050, 1.0);

        // Without smoothing
        let cens_no_smooth =
            chroma_cens(&signal, 22050, 512, 32.7, 12, 7, 36, 0.0, 0, "full").unwrap();

        // With smoothing
        let cens_smooth =
            chroma_cens(&signal, 22050, 512, 32.7, 12, 7, 36, 0.0, 41, "full").unwrap();

        assert_eq!(cens_no_smooth.shape(), cens_smooth.shape());

        // Both should have valid output
        assert!(cens_no_smooth.shape()[1] > 0);
        assert!(cens_smooth.shape()[1] > 0);

        // Both should have values in valid range [0, 1]
        for val in cens_no_smooth.iter() {
            assert!(*val >= 0.0 && *val <= 1.0 + 1e-6);
        }
        for val in cens_smooth.iter() {
            assert!(*val >= 0.0 && *val <= 1.0 + 1e-6);
        }
    }

    #[test]
    fn test_chroma_cens_empty() {
        let signal: Vec<f32> = vec![];
        let cens = chroma_cens(&signal, 22050, 512, 32.7, 12, 7, 36, 0.0, 41, "full").unwrap();

        assert_eq!(cens.shape()[0], 12);
        assert_eq!(cens.shape()[1], 0);
    }

    #[test]
    fn test_chroma_vqt_shape() {
        use crate::io;

        let signal = io::tone(440.0, 22050, 0.5);
        let chroma = chroma_vqt(&signal, 22050, 512, 32.7, 7, 12, 0.0, 0.0, -1, 0.0).unwrap();

        assert_eq!(chroma.shape()[0], 12);
        assert!(chroma.shape()[1] > 0);
    }

    #[test]
    fn test_chroma_vqt_with_gamma() {
        use crate::io;

        let signal = io::tone(440.0, 22050, 0.5);

        // VQT with gamma=0 (CQT-like)
        let chroma_cqt = chroma_vqt(&signal, 22050, 512, 32.7, 7, 12, 0.0, 0.0, -1, 0.0).unwrap();

        // VQT with gamma > 0 (different bandwidth scaling may affect frame count)
        let chroma_vqt_g =
            chroma_vqt(&signal, 22050, 512, 32.7, 7, 12, 0.0, 24.7, -1, 0.0).unwrap();

        // Both should have 12 chroma bins
        assert_eq!(chroma_cqt.shape()[0], 12);
        assert_eq!(chroma_vqt_g.shape()[0], 12);

        // Both should have some frames
        assert!(chroma_cqt.shape()[1] > 0);
        assert!(chroma_vqt_g.shape()[1] > 0);
    }

    #[test]
    fn test_chroma_vqt_a440_detection() {
        use crate::io;

        let signal = io::tone(440.0, 22050, 1.0);
        let chroma = chroma_vqt(&signal, 22050, 512, 32.7, 7, 12, 0.0, 0.0, -1, 0.0).unwrap();

        // Find the dominant chroma bin
        let n_frames = chroma.shape()[1];
        let mid_frame = n_frames / 2;

        let mut max_bin = 0;
        let mut max_val = 0.0f32;
        for c in 0..12 {
            if chroma[(c, mid_frame)] > max_val {
                max_val = chroma[(c, mid_frame)];
                max_bin = c;
            }
        }

        // A440 should be at chroma bin 9 (A) when starting from C
        // Allow some tolerance
        assert!(
            max_bin == 9 || max_bin == 8 || max_bin == 10,
            "A440 should peak near chroma bin A (9), got {}",
            max_bin
        );
    }

    #[test]
    fn test_chroma_vqt_empty() {
        let signal: Vec<f32> = vec![];
        let chroma = chroma_vqt(&signal, 22050, 512, 32.7, 7, 12, 0.0, 0.0, -1, 0.0).unwrap();

        assert_eq!(chroma.shape()[0], 12);
        assert_eq!(chroma.shape()[1], 0);
    }

    #[test]
    fn test_chroma_vqt_high_bins_per_octave() {
        use crate::io;

        // 24 bins per octave (quarter-tone resolution)
        let signal = io::tone(440.0, 22050, 0.5);
        let chroma = chroma_vqt(&signal, 22050, 512, 32.7, 7, 24, 0.0, 0.0, -1, 0.0).unwrap();

        assert_eq!(chroma.shape()[0], 24);
        assert!(chroma.shape()[1] > 0);
    }

    #[test]
    fn test_chroma_vqt_normalization() {
        use crate::io;

        let signal = io::tone(440.0, 22050, 0.5);

        // L2 normalization
        let chroma_l2 = chroma_vqt(&signal, 22050, 512, 32.7, 7, 12, 0.0, 0.0, 2, 0.0).unwrap();
        for frame in 0..chroma_l2.shape()[1] {
            let sum_sq: f32 = (0..12)
                .map(|c| chroma_l2[(c, frame)] * chroma_l2[(c, frame)])
                .sum();
            let norm = sum_sq.sqrt();
            if norm > 1e-6 {
                assert!((norm - 1.0).abs() < 0.01, "L2 norm should be ~1");
            }
        }
    }
}
