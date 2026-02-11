/// Convert frequencies (Hz) to mel scale.
///
/// # Arguments
/// * `frequencies` - Input frequencies in Hz
/// * `htk` - If true, use HTK formula; otherwise use Slaney formula (default: false)
///
/// # Example
/// ```
/// use giggle::convert::hz_to_mel;
///
/// let freqs = vec![0.0, 1000.0, 10000.0];
/// let mels = hz_to_mel(&freqs, false);
/// assert!(mels[1] > 0.0);
/// ```
pub fn hz_to_mel(frequencies: &[f32], htk: bool) -> Vec<f32> {
    if htk {
        // HTK formula
        frequencies
            .iter()
            .map(|&f| 2595.0 * (1.0 + f / 700.0).log10())
            .collect()
    } else {
        // Slaney formula
        const F_MIN: f32 = 0.0;
        const F_SP: f32 = 200.0 / 3.0;
        const MIN_LOG_HZ: f32 = 1000.0;
        const MIN_LOG_MEL: f32 = 15.0; // (MIN_LOG_HZ - F_MIN) / F_SP
        const LOGSTEP: f32 = 0.068751777; // log(6.4) / 27.0

        frequencies
            .iter()
            .map(|&f| {
                if f >= MIN_LOG_HZ {
                    MIN_LOG_MEL + (f / MIN_LOG_HZ).ln() / LOGSTEP
                } else {
                    (f - F_MIN) / F_SP
                }
            })
            .collect()
    }
}

/// Convert mel scale to frequencies (Hz).
///
/// # Arguments
/// * `mels` - Input mel values
/// * `htk` - If true, use HTK formula; otherwise use Slaney formula (default: false)
///
/// # Example
/// ```
/// use giggle::convert::{hz_to_mel, mel_to_hz};
///
/// let freqs = vec![1000.0];
/// let mels = hz_to_mel(&freqs, false);
/// let back = mel_to_hz(&mels, false);
/// assert!((back[0] - 1000.0).abs() < 1.0);
/// ```
pub fn mel_to_hz(mels: &[f32], htk: bool) -> Vec<f32> {
    if htk {
        // HTK formula
        mels.iter()
            .map(|&m| 700.0 * (10.0f32.powf(m / 2595.0) - 1.0))
            .collect()
    } else {
        // Slaney formula
        const F_MIN: f32 = 0.0;
        const F_SP: f32 = 200.0 / 3.0;
        const MIN_LOG_HZ: f32 = 1000.0;
        const MIN_LOG_MEL: f32 = 15.0;
        const LOGSTEP: f32 = 0.068751777;

        mels.iter()
            .map(|&m| {
                if m >= MIN_LOG_MEL {
                    MIN_LOG_HZ * ((m - MIN_LOG_MEL) * LOGSTEP).exp()
                } else {
                    F_MIN + F_SP * m
                }
            })
            .collect()
    }
}

/// Convert frequencies (Hz) to octaves relative to tuning.
///
/// # Arguments
/// * `frequencies` - Input frequencies in Hz
/// * `tuning` - Tuning offset in fractional bins (default: 0.0 for A440)
/// * `bins_per_octave` - Number of bins per octave (default: 12)
///
/// # Returns
/// Octave numbers where A440 is at octave 4.0
///
/// # Example
/// ```
/// use giggle::convert::hz_to_octs;
///
/// let freqs = vec![440.0, 880.0, 220.0];
/// let octs = hz_to_octs(&freqs, 0.0, 12);
/// assert!((octs[0] - 4.0).abs() < 0.01); // A440 = octave 4
/// assert!((octs[1] - 5.0).abs() < 0.01); // A880 = octave 5
/// ```
pub fn hz_to_octs(frequencies: &[f32], tuning: f32, bins_per_octave: usize) -> Vec<f32> {
    let a440 = 440.0 * 2.0_f32.powf(tuning / bins_per_octave as f32);
    let ref_freq = a440 / 16.0; // C0 reference

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

/// Convert octaves to frequencies (Hz).
///
/// # Arguments
/// * `octs` - Octave numbers
/// * `tuning` - Tuning offset in fractional bins (default: 0.0 for A440)
/// * `bins_per_octave` - Number of bins per octave (default: 12)
///
/// # Example
/// ```
/// use giggle::convert::{hz_to_octs, octs_to_hz};
///
/// let freqs = vec![440.0];
/// let octs = hz_to_octs(&freqs, 0.0, 12);
/// let back = octs_to_hz(&octs, 0.0, 12);
/// assert!((back[0] - 440.0).abs() < 0.01);
/// ```
pub fn octs_to_hz(octs: &[f32], tuning: f32, bins_per_octave: usize) -> Vec<f32> {
    let a440 = 440.0 * 2.0_f32.powf(tuning / bins_per_octave as f32);
    let ref_freq = a440 / 16.0;

    octs.iter().map(|&o| ref_freq * 2.0_f32.powf(o)).collect()
}
