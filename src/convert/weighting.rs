/// Compute A-weighting curve for frequencies.
///
/// A-weighting is a frequency response curve that approximates
/// the human ear's sensitivity to different frequencies.
///
/// # Arguments
/// * `frequencies` - Input frequencies in Hz
/// * `min_db` - Minimum weight in dB (values below this are set to min_db)
///
/// # Returns
/// Weight values in dB for each frequency
///
/// # Example
/// ```
/// use giggle::convert::a_weighting;
///
/// let freqs = vec![100.0, 1000.0, 10000.0];
/// let weights = a_weighting(&freqs, -80.0);
/// assert!(weights.len() == 3);
/// ```
pub fn a_weighting(frequencies: &[f32], min_db: f32) -> Vec<f32> {
    const F1: f32 = 20.6; // 20.6 Hz
    const F2: f32 = 107.7; // 107.7 Hz
    const F3: f32 = 737.9; // 737.9 Hz
    const F4: f32 = 12194.0; // 12194 Hz

    frequencies
        .iter()
        .map(|&f| {
            if f <= 0.0 {
                return min_db;
            }

            let f2 = f * f;
            let f4 = f2 * f2;

            let numerator = F4 * F4 * f4;
            let denominator =
                (f2 + F1 * F1) * ((f2 + F2 * F2) * (f2 + F3 * F3)).sqrt() * (f2 + F4 * F4);

            if denominator > 0.0 {
                let weight = 20.0 * (numerator / denominator).log10() + 2.0;
                weight.max(min_db)
            } else {
                min_db
            }
        })
        .collect()
}

/// Compute B-weighting curve for frequencies.
///
/// B-weighting is less commonly used but provides an intermediate
/// response between A and C weighting.
///
/// # Arguments
/// * `frequencies` - Input frequencies in Hz
/// * `min_db` - Minimum weight in dB
///
/// # Returns
/// Weight values in dB for each frequency
pub fn b_weighting(frequencies: &[f32], min_db: f32) -> Vec<f32> {
    const F1: f32 = 20.6;
    const F2: f32 = 158.5;
    const F4: f32 = 12194.0;

    frequencies
        .iter()
        .map(|&f| {
            if f <= 0.0 {
                return min_db;
            }

            let f2 = f * f;
            let f3 = f2 * f;

            let numerator = F4 * F4 * f3;
            let denominator = (f2 + F1 * F1) * (f2 + F2 * F2).sqrt() * (f2 + F4 * F4);

            if denominator > 0.0 {
                let weight = 20.0 * (numerator / denominator).log10() + 0.17;
                weight.max(min_db)
            } else {
                min_db
            }
        })
        .collect()
}

/// Compute C-weighting curve for frequencies.
///
/// C-weighting provides a relatively flat response and is used
/// for high-level sound measurements.
///
/// # Arguments
/// * `frequencies` - Input frequencies in Hz
/// * `min_db` - Minimum weight in dB
///
/// # Returns
/// Weight values in dB for each frequency
pub fn c_weighting(frequencies: &[f32], min_db: f32) -> Vec<f32> {
    const F1: f32 = 20.6;
    const F4: f32 = 12194.0;

    frequencies
        .iter()
        .map(|&f| {
            if f <= 0.0 {
                return min_db;
            }

            let f2 = f * f;

            let numerator = F4 * F4 * f2;
            let denominator = (f2 + F1 * F1) * (f2 + F4 * F4);

            if denominator > 0.0 {
                let weight = 20.0 * (numerator / denominator).log10() + 0.06;
                weight.max(min_db)
            } else {
                min_db
            }
        })
        .collect()
}

/// Compute Z-weighting (unweighted/flat) curve for frequencies.
///
/// Z-weighting provides flat frequency response (0 dB across all frequencies).
///
/// # Arguments
/// * `frequencies` - Input frequencies in Hz
///
/// # Returns
/// Zero weights for each frequency (flat response)
pub fn z_weighting(frequencies: &[f32]) -> Vec<f32> {
    vec![0.0; frequencies.len()]
}

/// Compute D-weighting curve for frequencies.
///
/// D-weighting is used for aircraft noise measurements.
///
/// # Arguments
/// * `frequencies` - Input frequencies in Hz
/// * `min_db` - Minimum weight in dB
///
/// # Returns
/// Weight values in dB for each frequency
pub fn d_weighting(frequencies: &[f32], min_db: f32) -> Vec<f32> {
    const _H1: f32 = 0.0;
    const _H2: f32 = 1.0;

    frequencies
        .iter()
        .map(|&f| {
            if f <= 0.0 {
                return min_db;
            }

            let f2 = f * f;

            // Simplified D-weighting (based on high-pass characteristics)
            let h_f = ((1_037_918.5 - f2).powi(2) + 1_080_768.1 * f2)
                / ((9837328.0 - f2).powi(2) + 11723776.0 * f2);

            let h_1000 = 0.456;
            let weight = 20.0 * (h_f / h_1000).log10();

            weight.max(min_db)
        })
        .collect()
}

/// Apply frequency weighting to a set of frequencies.
///
/// # Arguments
/// * `frequencies` - Input frequencies in Hz
/// * `kind` - Weighting type: "A", "B", "C", "D", or "Z"
/// * `min_db` - Minimum weight in dB (for A/B/C/D weighting)
///
/// # Returns
/// Weight values in dB for each frequency
///
/// # Example
/// ```
/// use giggle::convert::frequency_weighting;
///
/// let freqs = vec![1000.0];
/// let weights = frequency_weighting(&freqs, "A", -80.0);
/// assert!(weights[0].abs() < 5.0); // A-weighting is near 0 at 1kHz
/// ```
pub fn frequency_weighting(frequencies: &[f32], kind: &str, min_db: f32) -> Vec<f32> {
    match kind.to_uppercase().as_str() {
        "A" => a_weighting(frequencies, min_db),
        "B" => b_weighting(frequencies, min_db),
        "C" => c_weighting(frequencies, min_db),
        "D" => d_weighting(frequencies, min_db),
        "Z" => z_weighting(frequencies),
        _ => z_weighting(frequencies), // Default to flat
    }
}

/// Compute multiple frequency weightings simultaneously.
///
/// This function computes several frequency weighting curves at once,
/// returning a 2D array where each row corresponds to a different
/// weighting type.
///
/// # Arguments
/// * `frequencies` - Input frequencies in Hz
/// * `kinds` - Weighting types as a string (e.g., "ZAC") or slice of chars
/// * `min_db` - Minimum weight in dB (for A/B/C/D weighting)
///
/// # Returns
/// 2D array of shape (n_kinds, n_frequencies) with weight values in dB
///
/// # Example
/// ```
/// use giggle::convert::multi_frequency_weighting;
/// use ndarray::Array2;
///
/// let freqs = vec![100.0, 1000.0, 10000.0];
/// let weights = multi_frequency_weighting(&freqs, "ZAC", -80.0);
///
/// assert_eq!(weights.shape(), &[3, 3]); // 3 weighting types, 3 frequencies
///
/// // Z-weighting is flat (0 dB)
/// assert!((weights[(0, 0)] - 0.0).abs() < 0.01);
/// assert!((weights[(0, 1)] - 0.0).abs() < 0.01);
///
/// // A-weighting at 1kHz is close to 0
/// assert!(weights[(1, 1)].abs() < 1.0);
/// ```
pub fn multi_frequency_weighting(
    frequencies: &[f32],
    kinds: &str,
    min_db: f32,
) -> ndarray::Array2<f32> {
    use ndarray::Array2;

    let n_freqs = frequencies.len();
    let kinds_chars: Vec<char> = kinds.chars().collect();
    let n_kinds = kinds_chars.len();

    if n_kinds == 0 || n_freqs == 0 {
        return Array2::zeros((n_kinds, n_freqs));
    }

    let mut result = Array2::<f32>::zeros((n_kinds, n_freqs));

    for (i, kind) in kinds_chars.iter().enumerate() {
        let kind_str = kind.to_string();
        let weights = frequency_weighting(frequencies, &kind_str, min_db);
        for (j, &w) in weights.iter().enumerate() {
            result[(i, j)] = w;
        }
    }

    result
}

/// Compute multiple frequency weightings with explicit kind list.
///
/// Alternative to `multi_frequency_weighting` that takes a slice of
/// weighting kind strings.
///
/// # Arguments
/// * `frequencies` - Input frequencies in Hz
/// * `kinds` - Slice of weighting type strings (e.g., &["A", "B", "C"])
/// * `min_db` - Minimum weight in dB
///
/// # Returns
/// 2D array of shape (n_kinds, n_frequencies)
///
/// # Example
/// ```
/// use giggle::convert::multi_frequency_weighting_kinds;
///
/// let freqs = vec![1000.0, 4000.0];
/// let kinds = ["A", "C", "Z"];
/// let weights = multi_frequency_weighting_kinds(&freqs, &kinds, -80.0);
///
/// assert_eq!(weights.shape(), &[3, 2]);
/// ```
pub fn multi_frequency_weighting_kinds(
    frequencies: &[f32],
    kinds: &[&str],
    min_db: f32,
) -> ndarray::Array2<f32> {
    use ndarray::Array2;

    let n_freqs = frequencies.len();
    let n_kinds = kinds.len();

    if n_kinds == 0 || n_freqs == 0 {
        return Array2::zeros((n_kinds, n_freqs));
    }

    let mut result = Array2::<f32>::zeros((n_kinds, n_freqs));

    for (i, kind) in kinds.iter().enumerate() {
        let weights = frequency_weighting(frequencies, kind, min_db);
        for (j, &w) in weights.iter().enumerate() {
            result[(i, j)] = w;
        }
    }

    result
}
