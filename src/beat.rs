/// Beat tracking functionality.
use crate::feature::tempo::fourier_tempogram;
use crate::onset::strength::onset_strength;

/// Configuration for beat tracking.
///
/// This struct provides a builder pattern for configuring beat tracking parameters.
///
/// # Example
/// ```
/// use giggle::beat::BeatTrackConfig;
///
/// let config = BeatTrackConfig::new(22050)
///     .with_start_bpm(Some(120.0));
/// ```
#[derive(Debug, Clone)]
pub struct BeatTrackConfig {
    /// Sample rate
    pub sr: u32,
    /// Hop length for STFT
    pub hop_length: usize,
    /// Initial tempo estimate
    pub start_bpm: Option<f32>,
}

impl BeatTrackConfig {
    /// Create a new beat tracking configuration with defaults.
    ///
    /// # Arguments
    /// * `sr` - Sample rate
    pub fn new(sr: u32) -> Self {
        Self {
            sr,
            hop_length: 512,
            start_bpm: None,
        }
    }

    /// Set the hop length.
    pub fn with_hop_length(mut self, hop_length: usize) -> Self {
        self.hop_length = hop_length;
        self
    }

    /// Set the initial tempo estimate.
    pub fn with_start_bpm(mut self, start_bpm: Option<f32>) -> Self {
        self.start_bpm = start_bpm;
        self
    }

    /// Track beats with this configuration.
    ///
    /// # Arguments
    /// * `y` - Audio signal
    /// * `onset_envelope` - Pre-computed onset strength (optional)
    ///
    /// # Returns
    /// (tempo, beat_frames) - Estimated tempo in BPM and beat frame indices
    pub fn track(
        &self,
        y: &[f32],
        onset_envelope: Option<&[f32]>,
    ) -> crate::Result<(f32, Vec<usize>)> {
        beat_track(y, self.sr, onset_envelope, self.hop_length, self.start_bpm)
    }
}

impl Default for BeatTrackConfig {
    fn default() -> Self {
        Self {
            sr: 22050,
            hop_length: 512,
            start_bpm: None,
        }
    }
}

/// Simple beat tracking using onset strength and tempo.
///
/// # Arguments
/// * `y` - Audio signal
/// * `sr` - Sample rate
/// * `onset_envelope` - Pre-computed onset strength (optional)
/// * `hop_length` - Hop length for STFT
/// * `start_bpm` - Initial tempo estimate
///
/// # Returns
/// (tempo, beat_frames) - Estimated tempo in BPM and beat frame indices
pub fn beat_track(
    y: &[f32],
    sr: u32,
    onset_envelope: Option<&[f32]>,
    hop_length: usize,
    start_bpm: Option<f32>,
) -> crate::Result<(f32, Vec<usize>)> {
    // Get onset envelope
    let env = if let Some(e) = onset_envelope {
        e.to_vec()
    } else {
        onset_strength(y, 2048, hop_length)?
    };

    if env.len() < 2 {
        return Ok((120.0, Vec::new()));
    }

    // Estimate tempo using autocorrelation
    let tempo = estimate_tempo_from_onset(&env, sr, hop_length, start_bpm);

    // Find beats using dynamic programming
    let beats = find_beats(&env, tempo, sr, hop_length);

    Ok((tempo, beats))
}

/// Estimate tempo from onset envelope using autocorrelation.
fn estimate_tempo_from_onset(
    onset_env: &[f32],
    sr: u32,
    hop_length: usize,
    start_bpm: Option<f32>,
) -> f32 {
    if onset_env.len() < 3 {
        return start_bpm.unwrap_or(120.0);
    }

    // Autocorrelation for periodicity detection
    let max_lag = (sr as usize * 2 / hop_length).min(onset_env.len() / 2);
    let min_lag = (sr as usize / hop_length / 4).max(1); // Min 240 BPM

    let mut best_lag = min_lag;
    let mut best_score = 0.0f32;

    for lag in min_lag..max_lag {
        let mut score = 0.0f32;
        for i in 0..(onset_env.len() - lag) {
            score += onset_env[i] * onset_env[i + lag];
        }

        if score > best_score {
            best_score = score;
            best_lag = lag;
        }
    }

    // Convert lag to BPM
    let period_sec = best_lag as f32 * hop_length as f32 / sr as f32;
    let tempo = 60.0 / period_sec;

    // Clamp to reasonable range
    tempo.clamp(30.0, 300.0)
}

/// Predominant Local Pulse (PLP) estimation.
///
/// PLP analyzes the onset strength envelope in the frequency domain to find
/// locally stable tempo for each frame. These local periodicities are used
/// to synthesize local half-waves, which are combined such that peaks
/// coincide with rhythmically salient frames.
///
/// # Arguments
/// * `y` - Audio signal (optional if onset_envelope provided)
/// * `sr` - Sample rate
/// * `onset_envelope` - Pre-computed onset strength (optional)
/// * `hop_length` - Hop length for STFT
/// * `win_length` - Window length for tempogram analysis (default: 384)
/// * `tempo_min` - Minimum tempo in BPM (default: 30.0)
/// * `tempo_max` - Maximum tempo in BPM (default: 300.0)
///
/// # Returns
/// Pulse curve where maxima correspond to rhythmically salient points
///
/// # Example
/// ```
/// use giggle::beat::plp;
/// use giggle::io;
///
/// let signal = io::tone(1.0, 22050, 2.0); // 2 seconds
/// let pulse = plp(&signal, 22050, None, 512, 384, 30.0, 300.0).unwrap();
/// assert!(pulse.len() > 0);
/// ```
pub fn plp(
    y: &[f32],
    sr: u32,
    onset_envelope: Option<&[f32]>,
    hop_length: usize,
    win_length: usize,
    tempo_min: f32,
    tempo_max: f32,
) -> crate::Result<Vec<f32>> {
    // Step 1: Get onset envelope
    let env = if let Some(e) = onset_envelope {
        e.to_vec()
    } else {
        onset_strength(y, 2048, hop_length)?
    };

    if env.is_empty() || win_length == 0 {
        return Ok(Vec::new());
    }

    // Step 2: Compute Fourier tempogram (magnitude-only version)
    let ftgram = fourier_tempogram(&env, sr, hop_length, win_length);

    // Helper function to normalize a vector to [0, 1]
    fn normalize_vec(v: &mut [f32]) {
        let max_val = v.iter().copied().fold(0.0f32, f32::max);
        if max_val > 0.0 {
            for x in v.iter_mut() {
                *x /= max_val;
            }
        }
    }

    if ftgram.is_empty() {
        let mut result = env.clone();
        normalize_vec(&mut result);
        return Ok(result);
    }

    let n_tempo_bins = ftgram.shape()[0];
    let n_frames = ftgram.shape()[1];

    if n_tempo_bins == 0 || n_frames == 0 {
        let mut result = env.clone();
        normalize_vec(&mut result);
        return Ok(result);
    }

    // Step 3: Compute tempo frequencies
    let tempo_freqs: Vec<f32> = (0..n_tempo_bins)
        .map(|i| {
            let freq_hz = i as f32 * sr as f32 / (hop_length * win_length) as f32;
            freq_hz * 60.0 // Convert to BPM
        })
        .collect();

    // Step 4: Find predominant tempo at each frame and build pulse
    let mut pulse = vec![0.0f32; env.len()];

    for t in 0..n_frames {
        // Find peak tempo bin (within constraints)
        let mut max_mag = 0.0f32;
        let mut max_bin = 0usize;

        for bin in 0..n_tempo_bins {
            let tempo_bpm = tempo_freqs[bin];
            if tempo_bpm >= tempo_min && tempo_bpm <= tempo_max {
                let mag = ftgram[(bin, t)];
                if mag > max_mag {
                    max_mag = mag;
                    max_bin = bin;
                }
            }
        }

        // Build pulse based on predominant tempo
        if max_mag > 0.0 && max_bin > 0 {
            let tempo_bpm = tempo_freqs[max_bin];
            if tempo_bpm > 0.0 {
                let _period_frames = (60.0 * sr as f32) / (tempo_bpm * hop_length as f32);

                // Weight the onset envelope at this position by the tempo strength
                let frame_idx = t + win_length / 2; // Center the window
                if frame_idx < pulse.len() {
                    pulse[frame_idx] = env[frame_idx] * max_mag;
                }
            }
        }
    }

    // Normalize pulse to [0, 1]
    let max_pulse = pulse.iter().copied().fold(0.0f32, f32::max);
    if max_pulse > 0.0 {
        for p in &mut pulse {
            *p /= max_pulse;
        }
    }

    // Apply smoothing with simple moving average
    let smooth_window = 3;
    let mut smoothed = vec![0.0f32; pulse.len()];
    for (i, smoothed_val) in smoothed.iter_mut().enumerate().take(pulse.len()) {
        let start = i.saturating_sub(smooth_window);
        let end = (i + smooth_window + 1).min(pulse.len());
        let sum: f32 = pulse[start..end].iter().sum();
        let count = (end - start) as f32;
        *smoothed_val = if count > 0.0 { sum / count } else { 0.0 };
    }

    // Final normalization
    let max_smoothed = smoothed.iter().copied().fold(0.0f32, f32::max);
    if max_smoothed > 0.0 {
        for p in &mut smoothed {
            *p /= max_smoothed;
        }
    }

    Ok(smoothed)
}

/// Find beat positions using dynamic programming.
fn find_beats(onset_env: &[f32], tempo: f32, sr: u32, hop_length: usize) -> Vec<usize> {
    if onset_env.is_empty() {
        return Vec::new();
    }

    let period = (60.0 * sr as f32) / (tempo * hop_length as f32);
    let period_frames = period.round() as usize;

    if period_frames == 0 {
        return Vec::new();
    }

    // Simple peak picking at regular intervals
    let mut beats = Vec::new();
    let mut pos;

    // Find first peak
    let search_window = period_frames / 2;
    let mut max_pos = 0;
    let mut max_val = 0.0f32;

    for (i, &val) in onset_env
        .iter()
        .enumerate()
        .take(search_window.min(onset_env.len()))
    {
        if val > max_val {
            max_val = val;
            max_pos = i;
        }
    }

    beats.push(max_pos);
    pos = max_pos + period_frames;

    // Find subsequent beats
    while pos < onset_env.len() {
        let start = pos.saturating_sub(search_window);
        let end = (pos + search_window).min(onset_env.len());

        let mut local_max_pos = pos;
        let mut local_max_val = 0.0f32;

        for (i, &val) in onset_env.iter().enumerate().take(end).skip(start) {
            if val > local_max_val {
                local_max_val = val;
                local_max_pos = i;
            }
        }

        if local_max_val > onset_env.iter().copied().fold(0.0f32, f32::max) * 0.1 {
            beats.push(local_max_pos);
        }

        pos += period_frames;
    }

    beats
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_beat_track_basic() {
        // Create a simple signal with periodic impulses
        let sr = 22050;
        let hop_length = 512;

        // 120 BPM = 2 beats per second
        let beat_period_samples = sr / 2;
        let signal_length = sr * 4; // 4 seconds

        let mut signal = vec![0.0f32; signal_length as usize];
        let mut pos = 0;
        while pos < signal.len() {
            signal[pos] = 1.0;
            pos += beat_period_samples as usize;
        }

        let (tempo, beats) = beat_track(&signal, sr, None, hop_length, Some(120.0)).unwrap();

        // Should detect reasonable tempo
        assert!(tempo > 60.0 && tempo < 180.0);
        // Should find multiple beats
        assert!(beats.len() >= 2);
    }

    #[test]
    fn test_beat_track_empty() {
        let signal = vec![];
        let (tempo, beats) = beat_track(&signal, 22050, None, 512, None).unwrap();

        assert_eq!(tempo, 120.0);
        assert_eq!(beats.len(), 0);
    }

    #[test]
    fn test_estimate_tempo() {
        // Periodic onset envelope: 120 BPM (2 Hz) at 22050 Hz
        let sr = 22050u32;
        let hop_length = 512usize;
        let period_frames = sr as usize / hop_length / 2; // 2 Hz

        let mut env = vec![0.0f32; period_frames * 4];
        for i in 0..4 {
            env[i * period_frames] = 1.0;
        }

        let tempo = estimate_tempo_from_onset(&env, sr, hop_length, None);

        // Should be close to 120 BPM
        assert!(tempo > 60.0 && tempo < 180.0);
    }

    #[test]
    fn test_plp_basic() {
        // Create a simple signal
        let sr = 22050;
        let signal = vec![0.1f32; sr * 2]; // 2 seconds

        let pulse = super::plp(&signal, sr as u32, None, 512, 384, 30.0, 300.0).unwrap();

        // Should return a pulse curve
        assert!(!pulse.is_empty());

        // All values should be in [0, 1]
        for &p in &pulse {
            assert!((0.0..=1.0).contains(&p), "Pulse value {} out of range", p);
        }
    }

    #[test]
    fn test_plp_empty() {
        let signal: Vec<f32> = vec![];
        let pulse = super::plp(&signal, 22050, None, 512, 384, 30.0, 300.0).unwrap();
        assert_eq!(pulse.len(), 0);
    }

    #[test]
    fn test_plp_with_onset_envelope() {
        // Create a periodic onset envelope
        let mut env = vec![0.0f32; 200];
        for i in (0..200).step_by(20) {
            env[i] = 1.0;
        }

        let pulse = super::plp(&[], 22050, Some(&env), 512, 384, 30.0, 300.0).unwrap();

        assert!(!pulse.is_empty());
    }
}
