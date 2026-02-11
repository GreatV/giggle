use crate::onset::strength::onset_strength;

/// Configuration for onset detection.
///
/// This struct provides a builder pattern for configuring onset detection parameters.
///
/// # Example
/// ```
/// use giggle::onset::OnsetDetectConfig;
///
/// let config = OnsetDetectConfig::new()
///     .with_threshold(0.1);
/// ```
#[derive(Debug, Clone)]
pub struct OnsetDetectConfig {
    /// FFT window size
    pub n_fft: usize,
    /// Hop length for STFT
    pub hop_length: usize,
    /// Threshold for peak picking
    pub threshold: f32,
}

impl OnsetDetectConfig {
    /// Create a new onset detection configuration with defaults.
    pub fn new() -> Self {
        Self {
            n_fft: 2048,
            hop_length: 512,
            threshold: 0.1,
        }
    }

    /// Set the FFT window size.
    pub fn with_n_fft(mut self, n_fft: usize) -> Self {
        self.n_fft = n_fft;
        self
    }

    /// Set the hop length.
    pub fn with_hop_length(mut self, hop_length: usize) -> Self {
        self.hop_length = hop_length;
        self
    }

    /// Set the threshold for peak picking.
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    /// Detect onsets with this configuration.
    ///
    /// # Arguments
    /// * `y` - Input audio signal
    ///
    /// # Returns
    /// Vector of onset frame indices
    pub fn detect(&self, y: &[f32]) -> crate::Result<Vec<usize>> {
        onset_detect(y, self.n_fft, self.hop_length, self.threshold)
    }
}

impl Default for OnsetDetectConfig {
    fn default() -> Self {
        Self {
            n_fft: 2048,
            hop_length: 512,
            threshold: 0.1,
        }
    }
}

pub fn onset_detect(
    y: &[f32],
    n_fft: usize,
    hop_length: usize,
    threshold: f32,
) -> crate::Result<Vec<usize>> {
    let env = onset_strength(y, n_fft, hop_length)?;
    if env.is_empty() {
        return Ok(Vec::new());
    }

    let mut peaks = Vec::new();
    for i in 1..env.len().saturating_sub(1) {
        let v = env[i];
        if v > threshold && v >= env[i - 1] && v > env[i + 1] {
            peaks.push(i);
        }
    }
    Ok(peaks)
}

/// Backtrack detected onsets to the nearest energy minimum.
/// This refines onset positions by finding local energy minima.
///
/// # Arguments
/// * `onset_envelope` - Onset strength envelope
/// * `events` - Detected onset frame indices
/// * `search_range` - Maximum frames to search backward (default 3)
///
/// # Returns
/// Refined onset frame indices
pub fn onset_backtrack(
    onset_envelope: &[f32],
    events: &[usize],
    search_range: usize,
) -> Vec<usize> {
    if events.is_empty() || onset_envelope.is_empty() {
        return events.to_vec();
    }

    let mut refined = Vec::with_capacity(events.len());

    for &event in events {
        if event >= onset_envelope.len() {
            refined.push(event);
            continue;
        }

        // Search backward for local minimum
        let start = event.saturating_sub(search_range);
        let end = (event + 1).min(onset_envelope.len());

        let mut min_pos = event;
        let mut min_val = onset_envelope[event];

        for (i, &val) in onset_envelope.iter().enumerate().take(end).skip(start) {
            if val < min_val {
                min_val = val;
                min_pos = i;
            }
        }

        refined.push(min_pos);
    }

    refined
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_onset_backtrack_basic() {
        let envelope = vec![0.1, 0.3, 0.2, 0.5, 0.4, 0.2, 0.1];
        let events = vec![3]; // Peak at index 3 (value 0.5)

        let refined = onset_backtrack(&envelope, &events, 2);

        // Should backtrack to index 2 (value 0.2, local min before peak)
        assert_eq!(refined.len(), 1);
        assert_eq!(refined[0], 2);
    }

    #[test]
    fn test_onset_backtrack_multiple() {
        let envelope = vec![0.1, 0.5, 0.2, 0.1, 0.6, 0.3, 0.1];
        let events = vec![1, 4];

        let refined = onset_backtrack(&envelope, &events, 2);

        assert_eq!(refined.len(), 2);
        // First event should backtrack to 0
        assert_eq!(refined[0], 0);
        // Second event should backtrack to 3
        assert_eq!(refined[1], 3);
    }

    #[test]
    fn test_onset_backtrack_empty() {
        let envelope = vec![0.1, 0.2, 0.3];
        let events = vec![];

        let refined = onset_backtrack(&envelope, &events, 3);
        assert_eq!(refined.len(), 0);
    }

    #[test]
    fn test_onset_backtrack_no_change() {
        let envelope = vec![0.1, 0.2, 0.5, 0.4, 0.3];
        let events = vec![2]; // Already at local minimum before

        let refined = onset_backtrack(&envelope, &events, 1);
        assert_eq!(refined[0], 1); // Should find lower value at 1
    }
}
