//! Preemphasis and deemphasis filters for audio processing.

/// Apply preemphasis filter to emphasize high frequencies.
/// `y[n] = x[n] - coef * x[n-1]`
///
/// # Arguments
/// * `y` - Input audio signal
/// * `coef` - Preemphasis coefficient (typically 0.97)
/// * `zi` - Initial filter state (optional)
///
/// # Returns
/// Filtered signal and final filter state
pub fn preemphasis(y: &[f32], coef: f32, zi: Option<f32>) -> (Vec<f32>, f32) {
    if y.is_empty() {
        return (Vec::new(), 0.0);
    }

    let mut out = Vec::with_capacity(y.len());
    let mut prev = zi.unwrap_or(0.0);

    for &sample in y {
        let filtered = sample - coef * prev;
        out.push(filtered);
        prev = sample;
    }

    (out, prev)
}

/// Apply deemphasis filter (inverse of preemphasis).
/// `y[n] = x[n] + coef * y[n-1]`
///
/// # Arguments
/// * `y` - Input audio signal
/// * `coef` - Deemphasis coefficient (same as preemphasis)
/// * `zi` - Initial filter state (optional)
///
/// # Returns
/// Filtered signal and final filter state
pub fn deemphasis(y: &[f32], coef: f32, zi: Option<f32>) -> (Vec<f32>, f32) {
    if y.is_empty() {
        return (Vec::new(), 0.0);
    }

    let mut out = Vec::with_capacity(y.len());
    let mut prev = zi.unwrap_or(0.0);

    for &sample in y {
        let filtered = sample + coef * prev;
        out.push(filtered);
        prev = filtered;
    }

    (out, prev)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_preemphasis_basic() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (filtered, _) = preemphasis(&signal, 0.97, None);

        assert_eq!(filtered.len(), signal.len());
        assert_relative_eq!(filtered[0], 1.0, epsilon = 0.01);
        // Second sample: 2.0 - 0.97 * 1.0 = 1.03
        assert_relative_eq!(filtered[1], 1.03, epsilon = 0.01);
    }

    #[test]
    fn test_preemphasis_deemphasis_roundtrip() {
        let signal = vec![0.5, 1.0, 0.3, -0.2, 0.8, -0.5];
        let coef = 0.97;

        let (pre, _zi_pre) = preemphasis(&signal, coef, None);
        let (recovered, _) = deemphasis(&pre, coef, None);

        // Should approximately recover original signal
        // (slight error due to initial state)
        for i in 1..signal.len() {
            assert_relative_eq!(recovered[i], signal[i], epsilon = 0.01);
        }
    }

    #[test]
    fn test_preemphasis_with_state() {
        let signal = vec![1.0, 2.0, 3.0];
        let coef = 0.95;
        let zi = 0.5;

        let (filtered, final_state) = preemphasis(&signal, coef, Some(zi));

        // First output: 1.0 - 0.95 * 0.5 = 0.525
        assert_relative_eq!(filtered[0], 0.525, epsilon = 0.01);
        // Final state should be last input
        assert_relative_eq!(final_state, 3.0, epsilon = 0.01);
    }

    #[test]
    fn test_deemphasis_basic() {
        let signal = vec![1.0, 2.0, 3.0];
        let (filtered, _) = deemphasis(&signal, 0.95, None);

        assert_eq!(filtered.len(), signal.len());
        assert_relative_eq!(filtered[0], 1.0, epsilon = 0.01);
    }

    #[test]
    fn test_preemphasis_empty() {
        let signal = vec![];
        let (filtered, state) = preemphasis(&signal, 0.97, None);

        assert_eq!(filtered.len(), 0);
        assert_eq!(state, 0.0);
    }

    #[test]
    fn test_deemphasis_empty() {
        let signal = vec![];
        let (filtered, state) = deemphasis(&signal, 0.97, None);

        assert_eq!(filtered.len(), 0);
        assert_eq!(state, 0.0);
    }
}
