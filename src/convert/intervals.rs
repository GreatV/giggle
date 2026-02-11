use super::{fifths_to_note, hz_to_note};

/// Convert frequency (Hz) to Functional Just System (FJS) notation.
///
/// FJS is a notation system for just intonation that extends traditional
/// staff notation with superscript and subscript accidentals to indicate
/// otonal (overtone) and utonal (undertone) relationships.
///
/// # Arguments
/// * `freq` - Frequency in Hz
/// * `fmin` - Reference frequency (unison), if None uses freq as reference
/// * `unison` - Name of the unison note, if None infers from fmin
/// * `unicode` - If true, use unicode for accidentals and super/subscripts
///
/// # Returns
/// FJS notation string for the frequency
///
/// # Example
/// ```
/// use giggle::convert::hz_to_fjs;
///
/// // Simple cases
/// let note = hz_to_fjs(66.0, Some(55.0), None, true);
/// assert!(note.contains("C")); // 66/55 = 6/5, minor third from A
///
/// // Perfect fifth (3/2 ratio)
/// let fifth = hz_to_fjs(82.5, Some(55.0), None, true);
/// assert!(fifth.contains("E")); // 82.5/55 = 3/2
/// ```
pub fn hz_to_fjs(freq: f32, fmin: Option<f32>, unison: Option<&str>, unicode: bool) -> String {
    if freq <= 0.0 {
        return "?".to_string();
    }

    let fmin = fmin.unwrap_or(freq);
    if fmin <= 0.0 {
        return "?".to_string();
    }

    let unison = match unison {
        Some(u) => u.to_string(),
        None => {
            // Get note name without octave
            let full_note = hz_to_note(fmin);
            // Remove octave number (trailing digits)
            full_note.trim_end_matches(char::is_numeric).to_string()
        }
    };

    let interval = freq / fmin;
    interval_to_fjs(interval, &unison, unicode)
}

/// Convert multiple frequencies to FJS notation.
///
/// # Arguments
/// * `freqs` - Frequencies in Hz
/// * `fmin` - Reference frequency (if None, uses minimum of freqs)
/// * `unison` - Name of the unison note
/// * `unicode` - If true, use unicode symbols
///
/// # Returns
/// Vector of FJS notation strings
pub fn hz_to_fjs_batch(
    freqs: &[f32],
    fmin: Option<f32>,
    unison: Option<&str>,
    unicode: bool,
) -> Vec<String> {
    if freqs.is_empty() {
        return Vec::new();
    }

    let fmin = fmin.unwrap_or_else(|| freqs.iter().cloned().fold(f32::INFINITY, f32::min));

    freqs
        .iter()
        .map(|&f| hz_to_fjs(f, Some(fmin), unison, unicode))
        .collect()
}

/// Convert a just intonation interval ratio to FJS notation.
///
/// FJS notation represents intervals using:
/// - A base Pythagorean note name (derived from stacked perfect fifths)
/// - Superscript numbers for otonal (overtone series) factors
/// - Subscript numbers for utonal (undertone series) factors
///
/// # Arguments
/// * `interval` - The interval ratio (must be > 0)
/// * `unison` - The name of the unison note (e.g., "C", "A")
/// * `unicode` - If true, use unicode super/subscripts
///
/// # Returns
/// FJS notation string
///
/// # Example
/// ```
/// use giggle::convert::interval_to_fjs;
///
/// // Perfect fifth (3/2) from C
/// assert_eq!(interval_to_fjs(1.5, "C", true), "G");
///
/// // Perfect fourth (4/3) from C
/// assert_eq!(interval_to_fjs(4.0/3.0, "C", true), "F");
///
/// // Major third (5/4) from C - has otonal 5
/// let maj3 = interval_to_fjs(1.25, "C", true);
/// assert!(maj3.starts_with("E"));
///
/// // Unison
/// assert_eq!(interval_to_fjs(1.0, "A", true), "A");
/// ```
pub fn interval_to_fjs(interval: f32, unison: &str, unicode: bool) -> String {
    if interval <= 0.0 {
        return "?".to_string();
    }

    // Find the number of fifths to approximate this interval
    let fifths = fifth_search(interval, 65.0 / 63.0);

    // Get the base note name
    let note_name = fifths_to_note(unison, fifths, unicode);

    // Get prime factorization of the interval (octave-reduced)
    let interval_reduced = octave_fold(interval);
    let factors = get_interval_factors(interval_reduced);

    // Build otonal and utonal suffixes (ignoring factors of 2 and 3)
    let mut otonal: i64 = 1;
    let mut utonal: i64 = 1;

    for (&prime, &power) in &factors {
        if prime > 3 {
            if power > 0 {
                otonal *= (prime as i64).pow(power as u32);
            } else if power < 0 {
                utonal *= (prime as i64).pow((-power) as u32);
            }
        }
    }

    let mut suffix = String::new();

    if otonal > 1 {
        if unicode {
            suffix.push_str(&to_superscript(otonal));
        } else {
            suffix.push_str(&format!("^{}", otonal));
        }
    }

    if utonal > 1 {
        if unicode {
            suffix.push_str(&to_subscript(utonal));
        } else {
            suffix.push_str(&format!("_{}", utonal));
        }
    }

    format!("{}{}", note_name, suffix)
}

/// Fold an interval into the range [1, 2) by octave reduction.
fn octave_fold(interval: f32) -> f32 {
    if interval <= 0.0 {
        return 1.0;
    }
    let log2_int = interval.log2().floor();
    interval * 2.0_f32.powf(-log2_int)
}

/// Fold an interval into the balanced range [√2/2, √2) for comparison.
fn balanced_octave_fold(interval: f32) -> f32 {
    if interval <= 0.0 {
        return 1.0;
    }
    let log2_int = interval.log2().round();
    interval * 2.0_f32.powf(-log2_int)
}

/// Search for the number of perfect fifths that gets closest to the target interval.
fn fifth_search(interval: f32, tolerance: f32) -> i32 {
    let log_tolerance = tolerance.log2().abs();

    for power in 0..32 {
        for &sign in &[1i32, -1i32] {
            let fifths = power * sign;
            // 3/2 ratio for perfect fifth, but we're counting in Pythagorean terms
            let pythagorean = 3.0_f32.powi(fifths) / 2.0_f32.powi(fifths);
            let ratio = interval / pythagorean;
            let folded = balanced_octave_fold(ratio);

            if folded.log2().abs() <= log_tolerance {
                return fifths;
            }
        }
    }
    0
}

/// Get prime factorization of a just intonation interval.
/// Returns a map of prime -> power for common JI intervals.
fn get_interval_factors(interval: f32) -> std::collections::HashMap<i32, i32> {
    use std::collections::HashMap;

    // Table of common just intonation intervals (octave-reduced, rounded to 6 decimals)
    // Maps interval ratio -> prime factorization {prime: power}
    let ji_table: &[(f32, &[(i32, i32)])] = &[
        // Pythagorean intervals (only 2s and 3s)
        (1.0, &[]),             // unison
        (1.067871, &[(3, -5)]), // Pythagorean minor second (256/243)
        (1.125, &[(3, 2)]),     // Pythagorean major second (9/8)
        (1.185185, &[(3, -3)]), // Pythagorean minor third (32/27)
        (1.265625, &[(3, 4)]),  // Pythagorean major third (81/64)
        (1.333333, &[(3, -1)]), // Perfect fourth (4/3)
        (1.423828, &[(3, 6)]),  // Pythagorean tritone (729/512)
        (1.5, &[(3, 1)]),       // Perfect fifth (3/2)
        (1.580247, &[(3, -4)]), // Pythagorean minor sixth (128/81)
        (1.6875, &[(3, 3)]),    // Pythagorean major sixth (27/16)
        (1.777778, &[(3, -2)]), // Pythagorean minor seventh (16/9)
        (1.898438, &[(3, 5)]),  // Pythagorean major seventh (243/128)
        // 5-limit intervals
        (1.041667, &[(5, 1), (3, -2)]), // Syntonic comma (81/80) complement
        (1.066667, &[(5, -1), (3, 1)]), // Minor second 5-limit (16/15)
        (1.111111, &[(5, -1)]),         // Minor whole tone (10/9)
        (1.2, &[(5, 1), (3, -1)]),      // Minor third 5-limit (6/5)
        (1.25, &[(5, 1)]),              // Major third 5-limit (5/4)
        (1.6, &[(5, 1)]),               // Minor sixth 5-limit (8/5)
        (1.666667, &[(5, 1), (3, -1)]), // Major sixth 5-limit (5/3)
        (1.8, &[(5, -1), (3, 2)]),      // Minor seventh 5-limit (9/5)
        (1.875, &[(5, 1), (3, 1)]),     // Major seventh 5-limit (15/8)
        // 7-limit intervals
        (1.166667, &[(7, 1)]),          // Septimal minor third (7/6)
        (1.142857, &[(7, -1)]),         // Septimal whole tone (8/7)
        (1.285714, &[(7, -1), (3, 2)]), // Septimal major second (9/7)
        (1.4, &[(7, 1)]),               // Septimal tritone (7/5)
        (1.428571, &[(7, -1), (5, 1)]), // Septimal fifth (10/7)
        (1.75, &[(7, 1)]),              // Harmonic seventh (7/4)
        (1.714286, &[(7, -1), (3, 1)]), // Septimal major sixth (12/7)
    ];

    let mut result = HashMap::new();

    // Find closest match in table
    let mut best_match: Option<&[(i32, i32)]> = None;
    let mut best_diff = f32::INFINITY;

    for &(ratio, factors) in ji_table {
        let diff = (interval - ratio).abs();
        if diff < best_diff && diff < 0.001 {
            best_diff = diff;
            best_match = Some(factors);
        }
    }

    if let Some(factors) = best_match {
        for &(prime, power) in factors {
            result.insert(prime, power);
        }
    }

    result
}

/// Convert a number to unicode superscript.
fn to_superscript(n: i64) -> String {
    const SUPER: [char; 10] = ['⁰', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹'];
    n.to_string()
        .chars()
        .map(|c| {
            if let Some(digit) = c.to_digit(10) {
                SUPER[digit as usize]
            } else {
                c
            }
        })
        .collect()
}

/// Convert a number to unicode subscript.
fn to_subscript(n: i64) -> String {
    const SUB: [char; 10] = ['₀', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉'];
    n.to_string()
        .chars()
        .map(|c| {
            if let Some(digit) = c.to_digit(10) {
                SUB[digit as usize]
            } else {
                c
            }
        })
        .collect()
}

/// Helper function to parse a note name to pitch class (0-11).
pub(super) fn parse_note_to_pitch_class(note: &str) -> Option<i32> {
    let pitch_map: [(char, i32); 7] = [
        ('C', 0),
        ('D', 2),
        ('E', 4),
        ('F', 5),
        ('G', 7),
        ('A', 9),
        ('B', 11),
    ];

    let chars: Vec<char> = note.chars().collect();
    if chars.is_empty() {
        return None;
    }

    let base_note = chars[0].to_ascii_uppercase();
    let base_pitch = pitch_map
        .iter()
        .find(|(c, _)| *c == base_note)
        .map(|(_, p)| *p)?;

    let accidental_offset: i32 = chars[1..]
        .iter()
        .map(|&c| match c {
            '#' | '♯' => 1,
            'b' | '♭' | '!' => -1,
            _ => 0,
        })
        .sum();

    Some(((base_pitch + accidental_offset) % 12 + 12) % 12)
}

/// Construct a set of frequencies from an interval set.
///
/// # Arguments
/// * `n_bins` - The number of frequencies to generate
/// * `fmin` - The minimum frequency (must be > 0)
/// * `intervals` - Either a string specifying the interval type, or a vector of interval ratios
/// * `bins_per_octave` - Number of bins per octave (used when `intervals` is a string)
/// * `tuning` - Deviation from A440 tuning in fractional bins (only used for "equal" intervals)
/// * `sort` - Whether to sort the intervals in ascending order
///
/// # Interval Types
/// * `"equal"` - Equal temperament
/// * `"pythagorean"` - Pythagorean intervals
/// * `"ji3"` - 3-limit just intonation
/// * `"ji5"` - 5-limit just intonation
/// * `"ji7"` - 7-limit just intonation
/// * Custom intervals - Provide a vector of f32 ratios in range [1, 2)
///
/// # Returns
/// A vector of frequencies
///
/// # Examples
/// ```
/// use giggle::convert::interval_frequencies;
///
/// // Generate two octaves of Pythagorean intervals starting at 55Hz
/// let freqs = interval_frequencies(24, 55.0, "pythagorean", 12, 0.0, true);
/// assert_eq!(freqs.len(), 24);
/// ```
pub fn interval_frequencies(
    n_bins: usize,
    fmin: f32,
    intervals: &str,
    bins_per_octave: usize,
    tuning: f32,
    sort: bool,
) -> Vec<f32> {
    if fmin <= 0.0 || n_bins == 0 {
        return Vec::new();
    }

    let ratios: Vec<f32> = match intervals {
        "equal" => {
            // Equal temperament
            (0..bins_per_octave)
                .map(|i| 2.0f32.powf((tuning + i as f32) / bins_per_octave as f32))
                .collect()
        }
        "pythagorean" => pythagorean_intervals(bins_per_octave, sort, false),
        "ji3" => plimit_intervals(&[3], bins_per_octave, sort, false),
        "ji5" => plimit_intervals(&[3, 5], bins_per_octave, sort, false),
        "ji7" => plimit_intervals(&[3, 5, 7], bins_per_octave, sort, false),
        _ => return Vec::new(), // Unknown interval type
    };

    if ratios.is_empty() {
        return Vec::new();
    }

    let bpo = ratios.len();

    // Tile the ratios across octaves
    let n_octaves = (n_bins as f32 / bpo as f32).ceil() as usize;
    let mut all_ratios: Vec<f32> = Vec::with_capacity(n_octaves * bpo);

    for oct in 0..n_octaves {
        let octave_mult = 2.0f32.powi(oct as i32);
        for &r in &ratios {
            all_ratios.push(r * octave_mult);
        }
    }

    // Trim to n_bins
    all_ratios.truncate(n_bins);

    // Sort if requested
    if sort {
        all_ratios.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    }

    // Scale by fmin
    all_ratios.iter().map(|&r| r * fmin).collect()
}

/// Custom interval version of interval_frequencies.
///
/// # Arguments
/// * `n_bins` - The number of frequencies to generate
/// * `fmin` - The minimum frequency (must be > 0)
/// * `custom_intervals` - A vector of interval ratios in range [1, 2)
/// * `sort` - Whether to sort the intervals in ascending order
///
/// # Returns
/// A vector of frequencies
///
/// # Examples
/// ```
/// use giggle::convert::interval_frequencies_custom;
///
/// // Generate frequencies using just intonation intervals
/// let intervals = vec![1.0, 4.0/3.0, 3.0/2.0];
/// let freqs = interval_frequencies_custom(9, 55.0, &intervals, true);
/// assert_eq!(freqs.len(), 9);
/// ```
pub fn interval_frequencies_custom(
    n_bins: usize,
    fmin: f32,
    custom_intervals: &[f32],
    sort: bool,
) -> Vec<f32> {
    if fmin <= 0.0 || n_bins == 0 || custom_intervals.is_empty() {
        return Vec::new();
    }

    // Validate and filter intervals to [1, 2) range
    let ratios: Vec<f32> = custom_intervals
        .iter()
        .filter(|&&r| (1.0..2.0).contains(&r))
        .copied()
        .collect();

    if ratios.is_empty() {
        return Vec::new();
    }

    let bpo = ratios.len();

    // Tile the ratios across octaves
    let n_octaves = (n_bins as f32 / bpo as f32).ceil() as usize;
    let mut all_ratios: Vec<f32> = Vec::with_capacity(n_octaves * bpo);

    for oct in 0..n_octaves {
        let octave_mult = 2.0f32.powi(oct as i32);
        for &r in &ratios {
            all_ratios.push(r * octave_mult);
        }
    }

    // Trim to n_bins
    all_ratios.truncate(n_bins);

    // Sort if requested
    if sort {
        all_ratios.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    }

    // Scale by fmin
    all_ratios.iter().map(|&r| r * fmin).collect()
}

/// Pythagorean intervals.
///
/// Intervals are constructed by stacking ratios of 3/2 (perfect fifths)
/// and folding down to a single octave.
///
/// # Arguments
/// * `bins_per_octave` - The number of intervals to generate
/// * `sort` - If true, intervals are returned in ascending order.
///   If false, intervals are returned in circle-of-fifths order.
///
/// # Returns
/// A vector of interval ratios in range [1, 2)
///
/// # Examples
/// ```
/// use giggle::convert::pythagorean_intervals;
///
/// let intervals = pythagorean_intervals(12, true, false);
/// assert_eq!(intervals.len(), 12);
/// assert!((intervals[0] - 1.0).abs() < 1e-6);
/// ```
pub fn pythagorean_intervals(
    bins_per_octave: usize,
    sort: bool,
    _return_factors: bool,
) -> Vec<f32> {
    if bins_per_octave == 0 {
        return Vec::new();
    }

    // Generate all powers of 3 in log space
    let log3 = 3.0f32.log2();

    let mut log_ratios: Vec<(usize, f32, i32)> = Vec::with_capacity(bins_per_octave);

    for k in 0..bins_per_octave {
        // Calculate log2(3^k) = k * log2(3)
        let log_val = k as f32 * log3;

        // Get fractional part and power of 2
        let pow2 = log_val.floor() as i32;
        let mut frac = log_val - pow2 as f32;

        // Handle negative fractional parts
        if frac < 0.0 {
            frac += 1.0;
        }

        log_ratios.push((k, frac, pow2));
    }

    // Sort if requested
    if sort {
        log_ratios.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    }

    // Convert back to ratios
    log_ratios
        .into_iter()
        .take(bins_per_octave)
        .map(|(_, frac, _)| 2.0f32.powf(frac))
        .collect()
}

/// Compute harmonic distance between two ratios.
///
/// Harmonic distance is defined as log2(a * b) - 2*log2(gcd(a, b))
fn harmonic_distance(logs: &[f32], a: &[i32], b: &[i32]) -> f32 {
    // a_num = max(a, 0), a_den = max(-a, 0)
    let a_num: Vec<i32> = a.iter().map(|&x| x.max(0)).collect();
    let a_den: Vec<i32> = a.iter().map(|&x| (-x).max(0)).collect();

    let b_num: Vec<i32> = b.iter().map(|&x| x.max(0)).collect();
    let b_den: Vec<i32> = b.iter().map(|&x| (-x).max(0)).collect();

    // gcd = min(a_num, b_num) - max(a_den, b_den)
    let gcd: Vec<i32> = a_num
        .iter()
        .zip(b_num.iter())
        .zip(a_den.iter().zip(b_den.iter()))
        .map(|((an, bn), (ad, bd))| (*an).min(*bn) - (*ad).max(*bd))
        .collect();

    // HD = logs.dot(a + b - 2 * gcd)
    let sum: f32 = a
        .iter()
        .zip(b.iter())
        .zip(gcd.iter())
        .zip(logs.iter())
        .map(|(((ai, bi), gi), li)| li * (ai + bi - 2 * gi) as f32)
        .sum();

    // Round to 6 decimals to avoid floating point issues
    (sum * 1_000_000.0).round() / 1_000_000.0
}

/// Break ties in crystal growth algorithm.
fn crystal_tie_break(a: &[i32], b: &[i32], logs: &[f32]) -> bool {
    let a_sum: f32 = a
        .iter()
        .map(|&x| x.abs() as f32)
        .zip(logs.iter())
        .map(|(x, l)| x * l)
        .sum();
    let b_sum: f32 = b
        .iter()
        .map(|&x| x.abs() as f32)
        .zip(logs.iter())
        .map(|(x, l)| x * l)
        .sum();
    a_sum < b_sum
}

/// Construct p-limit intervals for a given set of prime factors.
///
/// This function implements the "harmonic crystal growth" algorithm
/// based on Tenney's work.
///
/// # Arguments
/// * `primes` - Slice of odd prime numbers to use
/// * `bins_per_octave` - The number of intervals to construct
/// * `sort` - If true, intervals are returned in ascending order.
///   If false, intervals are returned in crystal growth order.
/// * `return_factors` - If true, returns empty (factors not implemented in this variant)
///
/// # Returns
/// A vector of interval ratios in range [1, 2)
///
/// # Examples
/// ```
/// use giggle::convert::plimit_intervals;
///
/// // 5-limit just intonation
/// let intervals = plimit_intervals(&[3, 5], 12, true, false);
/// assert_eq!(intervals.len(), 12);
/// ```
pub fn plimit_intervals(
    primes: &[i32],
    bins_per_octave: usize,
    sort: bool,
    _return_factors: bool,
) -> Vec<f32> {
    if primes.is_empty() || bins_per_octave == 0 {
        return Vec::new();
    }

    // Filter to only odd primes
    let primes: Vec<i32> = primes.iter().filter(|&&p| p > 2).copied().collect();
    if primes.is_empty() {
        return Vec::new();
    }

    let n_primes = primes.len();

    // Precompute logs of primes
    let logs: Vec<f32> = primes.iter().map(|&p| (p as f32).log2()).collect();

    // Create seeds (primes and their reciprocals)
    let mut seeds: Vec<Vec<i32>> = Vec::new();
    for i in 0..n_primes {
        let mut seed = vec![0i32; n_primes];
        seed[i] = 1;
        seeds.push(seed.clone());
        seed[i] = -1;
        seeds.push(seed);
    }

    // Frontier: candidate intervals for inclusion
    let mut frontier: Vec<Vec<i32>> = seeds.clone();

    // Distances table
    let mut distances: std::collections::HashMap<(Vec<i32>, Vec<i32>), f32> =
        std::collections::HashMap::new();

    // Initialize with root (1)
    let mut intervals: Vec<Vec<i32>> = Vec::new();
    let root = vec![0i32; n_primes];
    intervals.push(root);

    while intervals.len() < bins_per_octave {
        let mut best_score = f32::INFINITY;
        let mut best_idx = 0;

        for (f_idx, point) in frontier.iter().enumerate() {
            // Compute harmonic distance to each selected interval
            let mut hd = 0.0f32;

            for s in &intervals {
                let key = (s.clone(), point.clone());
                let dist = if let Some(&d) = distances.get(&key) {
                    d
                } else {
                    let d = harmonic_distance(&logs, point, s);
                    distances.insert(key.clone(), d);
                    distances.insert((point.clone(), s.clone()), d);
                    d
                };
                hd += dist;
            }

            if hd < best_score
                || (hd == best_score && crystal_tie_break(point, &frontier[best_idx], &logs))
            {
                best_score = hd;
                best_idx = f_idx;
            }
        }

        let new_point = frontier.remove(best_idx);
        intervals.push(new_point.clone());

        // Add new seeds to frontier
        for seed in &seeds {
            let new_seed: Vec<i32> = new_point
                .iter()
                .zip(seed.iter())
                .map(|(a, b)| a + b)
                .collect();
            if !intervals.contains(&new_seed) && !frontier.contains(&new_seed) {
                frontier.push(new_seed);
            }
        }
    }

    // Convert to log ratios and powers of 2
    let mut log_ratios: Vec<(usize, f32, i32)> = Vec::with_capacity(bins_per_octave);

    for (idx, pows) in intervals.iter().enumerate() {
        // Compute dot product of powers with logs
        let log_val: f32 = pows
            .iter()
            .zip(logs.iter())
            .map(|(p, l)| *p as f32 * l)
            .sum();

        // Get fractional part and power of 2
        let mut pow2 = log_val.floor() as i32;
        let mut frac = log_val - pow2 as f32;

        // Handle negative fractional parts
        if frac < 0.0 {
            frac += 1.0;
            pow2 -= 1;
        }

        log_ratios.push((idx, frac, pow2));
    }

    // Sort if requested
    if sort {
        log_ratios.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    }

    // Convert back to ratios
    log_ratios
        .into_iter()
        .take(bins_per_octave)
        .map(|(_, frac, _)| 2.0f32.powf(frac))
        .collect()
}
