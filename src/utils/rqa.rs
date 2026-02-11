use ndarray::Array2;

/// Recurrence Quantification Analysis (RQA) metrics.
///
/// Computes statistics from a recurrence matrix to characterize
/// complex patterns and dynamics in time series data.
///
/// # Arguments
/// * `recurrence` - Binary recurrence matrix (n_frames x n_frames)
///
/// # Returns
/// Tuple of (determinism, laminarity, entropy) where:
/// - determinism: Ratio of recurrence points in diagonal structures
/// - laminarity: Ratio of recurrence points in vertical structures
/// - entropy: Shannon entropy of diagonal line length distribution
///
/// # Example
/// ```
/// use giggle::utils::{recurrence_matrix, rqa};
/// use ndarray::Array2;
///
/// let features = Array2::from_shape_vec((2, 8), (0..16).map(|x| x as f32).collect()).unwrap();
/// let rec = recurrence_matrix(&features, "connectivity", "cosine", 0.9);
/// let (det, lam, ent) = rqa(&rec);
/// assert!(det >= 0.0 && det <= 1.0);
/// ```
pub fn rqa(recurrence: &Array2<f32>) -> (f32, f32, f32) {
    let n = recurrence.shape()[0];
    if n == 0 || recurrence.shape()[1] != n {
        return (0.0, 0.0, 0.0);
    }

    let mut total_points = 0;
    let mut diagonal_points = 0;
    let mut vertical_points = 0;
    let mut diagonal_lengths = std::collections::HashMap::new();

    // Count total recurrence points
    for i in 0..n {
        for j in 0..n {
            if recurrence[(i, j)] > 0.5 {
                total_points += 1;
            }
        }
    }

    if total_points == 0 {
        return (0.0, 0.0, 0.0);
    }

    // Analyze diagonal lines (excluding main diagonal)
    for offset in 1..n {
        let mut length = 0;
        for i in 0..(n - offset) {
            let j = i + offset;
            if recurrence[(i, j)] > 0.5 {
                length += 1;
            } else {
                if length >= 2 {
                    // Diagonal line of at least length 2
                    *diagonal_lengths.entry(length).or_insert(0) += 1;
                    diagonal_points += length;
                }
                length = 0;
            }
        }
        // Handle line extending to edge
        if length >= 2 {
            *diagonal_lengths.entry(length).or_insert(0) += 1;
            diagonal_points += length;
        }
    }

    // Also check below diagonal
    for offset in 1..n {
        let mut length = 0;
        for i in offset..n {
            let j = i - offset;
            if recurrence[(i, j)] > 0.5 {
                length += 1;
            } else {
                if length >= 2 {
                    *diagonal_lengths.entry(length).or_insert(0) += 1;
                    diagonal_points += length;
                }
                length = 0;
            }
        }
        if length >= 2 {
            *diagonal_lengths.entry(length).or_insert(0) += 1;
            diagonal_points += length;
        }
    }

    // Analyze vertical lines (laminarity)
    for j in 0..n {
        let mut length = 0;
        for i in 0..n {
            if recurrence[(i, j)] > 0.5 {
                length += 1;
            } else {
                if length >= 2 {
                    vertical_points += length;
                }
                length = 0;
            }
        }
        if length >= 2 {
            vertical_points += length;
        }
    }

    // Compute metrics
    let determinism = diagonal_points as f32 / total_points as f32;
    let laminarity = vertical_points as f32 / total_points as f32;

    // Compute Shannon entropy of diagonal length distribution
    let mut entropy = 0.0f32;
    if !diagonal_lengths.is_empty() {
        let total_diagonals: usize = diagonal_lengths.values().sum();
        for &count in diagonal_lengths.values() {
            let p = count as f32 / total_diagonals as f32;
            if p > 0.0 {
                entropy -= p * p.log2();
            }
        }
    }

    (determinism, laminarity, entropy)
}

/// Compute RQA metrics with configurable minimum line length.
///
/// # Arguments
/// * `recurrence` - Binary recurrence matrix
/// * `min_length` - Minimum line length to consider (default: 2)
///
/// # Returns
/// Tuple of (determinism, laminarity, entropy, avg_diagonal_length)
pub fn rqa_detailed(recurrence: &Array2<f32>, min_length: usize) -> (f32, f32, f32, f32) {
    let n = recurrence.shape()[0];
    if n == 0 || recurrence.shape()[1] != n {
        return (0.0, 0.0, 0.0, 0.0);
    }

    let mut total_points = 0;
    let mut diagonal_points = 0;
    let mut vertical_points = 0;
    let mut diagonal_lengths = Vec::new();

    // Count total recurrence points
    for i in 0..n {
        for j in 0..n {
            if recurrence[(i, j)] > 0.5 {
                total_points += 1;
            }
        }
    }

    if total_points == 0 {
        return (0.0, 0.0, 0.0, 0.0);
    }

    // Analyze all diagonals (both upper and lower)
    for offset in -(n as i32 - 1)..=(n as i32 - 1) {
        let mut length = 0;
        let start_i = if offset >= 0 { 0 } else { -offset as usize };
        let start_j = if offset >= 0 { offset as usize } else { 0 };

        let mut i = start_i;
        let mut j = start_j;

        while i < n && j < n {
            if recurrence[(i, j)] > 0.5 {
                length += 1;
            } else {
                if length >= min_length {
                    diagonal_lengths.push(length);
                    diagonal_points += length;
                }
                length = 0;
            }
            i += 1;
            j += 1;
        }
        if length >= min_length {
            diagonal_lengths.push(length);
            diagonal_points += length;
        }
    }

    // Analyze vertical lines
    for j in 0..n {
        let mut length = 0;
        for i in 0..n {
            if recurrence[(i, j)] > 0.5 {
                length += 1;
            } else {
                if length >= min_length {
                    vertical_points += length;
                }
                length = 0;
            }
        }
        if length >= min_length {
            vertical_points += length;
        }
    }

    // Compute metrics
    let determinism = diagonal_points as f32 / total_points as f32;
    let laminarity = vertical_points as f32 / total_points as f32;

    let avg_diagonal_length = if !diagonal_lengths.is_empty() {
        diagonal_lengths.iter().sum::<usize>() as f32 / diagonal_lengths.len() as f32
    } else {
        0.0
    };

    // Shannon entropy
    let mut entropy = 0.0f32;
    if !diagonal_lengths.is_empty() {
        let mut length_counts = std::collections::HashMap::new();
        for &len in &diagonal_lengths {
            *length_counts.entry(len).or_insert(0) += 1;
        }

        let total = diagonal_lengths.len() as f32;
        for &count in length_counts.values() {
            let p = count as f32 / total;
            if p > 0.0 {
                entropy -= p * p.log2();
            }
        }
    }

    (determinism, laminarity, entropy, avg_diagonal_length)
}
