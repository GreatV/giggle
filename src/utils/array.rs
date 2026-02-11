use ndarray::Array2;

/// Normalize an array to have unit norm.
pub fn normalize(x: &[f32], norm: NormType) -> Vec<f32> {
    if x.is_empty() {
        return Vec::new();
    }

    let norm_val = match norm {
        NormType::L1 => x.iter().map(|v| v.abs()).sum::<f32>(),
        NormType::L2 => x.iter().map(|v| v * v).sum::<f32>().sqrt(),
        NormType::Max => x.iter().map(|v| v.abs()).fold(0.0f32, f32::max),
    };

    if norm_val > 1e-10 {
        x.iter().map(|v| v / norm_val).collect()
    } else {
        x.to_vec()
    }
}

/// Normalize a 2D array along axis.
pub fn normalize_2d(x: &Array2<f32>, norm: NormType, axis: usize) -> Array2<f32> {
    let mut result = x.clone();
    let shape = x.shape();

    if axis == 0 {
        for col in 0..shape[1] {
            let col_data: Vec<f32> = (0..shape[0]).map(|row| x[(row, col)]).collect();
            let normalized = normalize(&col_data, norm);
            for (row, &val) in normalized.iter().enumerate() {
                result[(row, col)] = val;
            }
        }
    } else {
        for row in 0..shape[0] {
            let row_data: Vec<f32> = (0..shape[1]).map(|col| x[(row, col)]).collect();
            let normalized = normalize(&row_data, norm);
            for (col, &val) in normalized.iter().enumerate() {
                result[(row, col)] = val;
            }
        }
    }

    result
}

/// Find local maxima in a signal.
pub fn localmax(x: &[f32]) -> Vec<usize> {
    let mut peaks = Vec::new();
    if x.len() < 3 {
        return peaks;
    }

    for i in 1..x.len() - 1 {
        if x[i] > x[i - 1] && x[i] > x[i + 1] {
            peaks.push(i);
        }
    }
    peaks
}

/// Find local minima in a 1D array.
///
/// An element `x[i]` is considered a local minimum if:
/// - `x[i] < x[i-1]` (strict)
/// - `x[i] <= x[i+1]`
///
/// Note: The first element is never considered a local minimum.
///
/// # Arguments
/// * `x` - Input array
///
/// # Returns
/// Boolean vector indicating local minima positions
///
/// # Example
/// ```
/// use giggle::utils::localmin;
///
/// let x = vec![1.0, 0.0, 1.0, 2.0, -1.0, 0.0, -2.0, 1.0];
/// let mins = localmin(&x);
/// assert_eq!(mins, vec![false, true, false, false, true, false, true, false]);
/// ```
pub fn localmin(x: &[f32]) -> Vec<bool> {
    let n = x.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![false];
    }

    let mut result = vec![false; n];

    // Check interior points
    for i in 1..n - 1 {
        result[i] = x[i] < x[i - 1] && x[i] <= x[i + 1];
    }

    // Handle the last element: it's a local minimum if strictly less than previous
    result[n - 1] = x[n - 1] < x[n - 2];

    result
}

/// Find local minima in a 2D array along a specified axis.
///
/// # Arguments
/// * `x` - Input 2D array
/// * `axis` - Axis along which to compute local minimality (0 = columns, 1 = rows)
///
/// # Returns
/// Boolean array indicating local minima positions
///
/// # Example
/// ```
/// use giggle::utils::localmin_2d;
/// use ndarray::Array2;
///
/// let x = Array2::from_shape_vec((3, 3), vec![
///     1.0, 0.0, 1.0,
///     2.0, -1.0, 0.0,
///     2.0, 1.0, 3.0,
/// ]).unwrap();
///
/// let mins = localmin_2d(&x, 0);
/// // Along axis 0 (down columns)
/// assert_eq!(mins.shape(), &[3, 3]);
/// ```
pub fn localmin_2d(x: &Array2<f32>, axis: usize) -> Array2<bool> {
    let (n_rows, n_cols) = (x.shape()[0], x.shape()[1]);

    if n_rows == 0 || n_cols == 0 {
        return Array2::from_elem((n_rows, n_cols), false);
    }

    let mut result = Array2::from_elem((n_rows, n_cols), false);

    if axis == 0 {
        // Along columns (compare rows)
        for j in 0..n_cols {
            if n_rows < 2 {
                continue;
            }
            for i in 1..n_rows - 1 {
                result[(i, j)] = x[(i, j)] < x[(i - 1, j)] && x[(i, j)] <= x[(i + 1, j)];
            }
            // Last row
            result[(n_rows - 1, j)] = x[(n_rows - 1, j)] < x[(n_rows - 2, j)];
        }
    } else {
        // Along rows (compare columns)
        for i in 0..n_rows {
            if n_cols < 2 {
                continue;
            }
            for j in 1..n_cols - 1 {
                result[(i, j)] = x[(i, j)] < x[(i, j - 1)] && x[(i, j)] <= x[(i, j + 1)];
            }
            // Last column
            result[(i, n_cols - 1)] = x[(i, n_cols - 1)] < x[(i, n_cols - 2)];
        }
    }

    result
}

/// Sort a 2D array along rows or columns by peak position.
///
/// # Arguments
/// * `s` - Input 2D array
/// * `axis` - Axis along which to sort:
///   - `0`: Sort rows by peak column index
///   - `1` or `-1`: Sort columns by peak row index
/// * `by_argmin` - If true, sort by minimum position instead of maximum
///
/// # Returns
/// Tuple of (sorted array, sorting indices)
///
/// # Example
/// ```
/// use giggle::utils::axis_sort;
/// use ndarray::Array2;
///
/// let s = Array2::from_shape_vec((3, 4), vec![
///     0.0, 1.0, 2.0, 0.0,  // peak at col 2
///     1.0, 0.0, 0.0, 0.0,  // peak at col 0
///     0.0, 0.0, 0.0, 3.0,  // peak at col 3
/// ]).unwrap();
///
/// let (sorted, idx) = axis_sort(&s, 0, false);
/// // Rows reordered by peak column: row1, row0, row2
/// assert_eq!(idx, vec![1, 0, 2]);
/// ```
pub fn axis_sort(s: &Array2<f32>, axis: i32, by_argmin: bool) -> (Array2<f32>, Vec<usize>) {
    let (n_rows, n_cols) = (s.shape()[0], s.shape()[1]);

    if n_rows == 0 || n_cols == 0 {
        return (s.clone(), Vec::new());
    }

    // Normalize axis
    let axis = if axis < 0 {
        (2 + axis) as usize
    } else {
        axis as usize
    };

    // Compute the index for each row/column
    let bin_idx: Vec<usize> = if axis == 0 {
        // Sort rows by their peak column
        (0..n_rows)
            .map(|i| {
                let row: Vec<f32> = (0..n_cols).map(|j| s[(i, j)]).collect();
                if by_argmin {
                    row.iter()
                        .enumerate()
                        .min_by(|(_, a), (_, b)| {
                            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|(idx, _)| idx)
                        .unwrap_or(0)
                } else {
                    row.iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| {
                            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|(idx, _)| idx)
                        .unwrap_or(0)
                }
            })
            .collect()
    } else {
        // Sort columns by their peak row
        (0..n_cols)
            .map(|j| {
                let col: Vec<f32> = (0..n_rows).map(|i| s[(i, j)]).collect();
                if by_argmin {
                    col.iter()
                        .enumerate()
                        .min_by(|(_, a), (_, b)| {
                            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|(idx, _)| idx)
                        .unwrap_or(0)
                } else {
                    col.iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| {
                            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|(idx, _)| idx)
                        .unwrap_or(0)
                }
            })
            .collect()
    };

    // Get sorting order
    let mut idx: Vec<usize> = (0..bin_idx.len()).collect();
    idx.sort_by_key(|&i| bin_idx[i]);

    // Apply permutation
    let sorted = if axis == 0 {
        let mut result = Array2::<f32>::zeros((n_rows, n_cols));
        for (new_i, &old_i) in idx.iter().enumerate() {
            for j in 0..n_cols {
                result[(new_i, j)] = s[(old_i, j)];
            }
        }
        result
    } else {
        let mut result = Array2::<f32>::zeros((n_rows, n_cols));
        for (new_j, &old_j) in idx.iter().enumerate() {
            for i in 0..n_rows {
                result[(i, new_j)] = s[(i, old_j)];
            }
        }
        result
    };

    (sorted, idx)
}

/// Sparse row representation of a matrix.
///
/// Represents a row-sparse matrix where small values are zeroed out.
#[derive(Debug, Clone)]
pub struct SparseRows {
    /// Number of rows
    pub n_rows: usize,
    /// Number of columns
    pub n_cols: usize,
    /// Row indices for each non-zero element
    pub row_indices: Vec<usize>,
    /// Column indices for each non-zero element
    pub col_indices: Vec<usize>,
    /// Non-zero values
    pub values: Vec<f32>,
}

impl SparseRows {
    /// Convert sparse representation back to dense array.
    pub fn to_dense(&self) -> Array2<f32> {
        let mut result = Array2::<f32>::zeros((self.n_rows, self.n_cols));
        for ((&row, &col), &val) in self
            .row_indices
            .iter()
            .zip(self.col_indices.iter())
            .zip(self.values.iter())
        {
            result[(row, col)] = val;
        }
        result
    }

    /// Get number of non-zero elements.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }
}

/// Return a row-sparse matrix approximating the input.
///
/// For each row, discard elements whose cumulative magnitude is below
/// the specified quantile.
///
/// # Arguments
/// * `x` - Input array (1D or 2D)
/// * `quantile` - Percentage of magnitude to discard in each row (0.0 to 1.0)
///
/// # Returns
/// Sparse row representation
///
/// # Example
/// ```
/// use giggle::utils::sparsify_rows;
/// use ndarray::Array2;
///
/// let x = Array2::from_shape_vec((1, 8), vec![
///     0.0, 0.1, 0.3, 0.5, 0.8, 0.5, 0.3, 0.1,
/// ]).unwrap();
///
/// let sparse = sparsify_rows(&x, 0.1);
/// // Small values at the edges should be zeroed
/// assert!(sparse.nnz() < 8);
/// ```
pub fn sparsify_rows(x: &Array2<f32>, quantile: f32) -> SparseRows {
    let (n_rows, n_cols) = (x.shape()[0], x.shape()[1]);

    if n_rows == 0 || n_cols == 0 || !(0.0..1.0).contains(&quantile) {
        return SparseRows {
            n_rows,
            n_cols,
            row_indices: Vec::new(),
            col_indices: Vec::new(),
            values: Vec::new(),
        };
    }

    let mut row_indices = Vec::new();
    let mut col_indices = Vec::new();
    let mut values = Vec::new();

    for i in 0..n_rows {
        // Get magnitudes for this row
        let mut mags: Vec<(usize, f32)> = (0..n_cols).map(|j| (j, x[(i, j)].abs())).collect();

        // Compute row norm
        let norm: f32 = mags.iter().map(|(_, m)| m).sum();
        if norm < 1e-10 {
            continue;
        }

        // Sort by magnitude
        mags.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Compute cumulative sum
        let mut cumsum = 0.0f32;
        let mut threshold_idx = 0;
        for (idx, (_, mag)) in mags.iter().enumerate() {
            cumsum += mag / norm;
            if cumsum >= quantile {
                threshold_idx = idx;
                break;
            }
        }

        // Get threshold magnitude
        let threshold_mag = mags[threshold_idx].1;

        // Keep elements with magnitude >= threshold
        for j in 0..n_cols {
            if x[(i, j)].abs() >= threshold_mag {
                row_indices.push(i);
                col_indices.push(j);
                values.push(x[(i, j)]);
            }
        }
    }

    SparseRows {
        n_rows,
        n_cols,
        row_indices,
        col_indices,
        values,
    }
}

/// Aggregate a 2D array between specified boundaries.
///
/// Useful for beat-synchronous feature aggregation.
///
/// # Arguments
/// * `data` - 2D feature array (n_features x n_frames)
/// * `idx` - Boundary indices (sorted)
/// * `aggregate` - Aggregation mode: "mean", "median", "min", "max"
/// * `pad` - If true, pad indices to span full range [0, n_frames]
///
/// # Returns
/// Aggregated array (n_features x n_segments)
///
/// # Example
/// ```
/// use giggle::utils::sync;
/// use ndarray::Array2;
///
/// let data = Array2::from_shape_vec((2, 10), vec![
///     1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
///     0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
/// ]).unwrap();
///
/// // Aggregate into 2 segments: [0,5) and [5,10)
/// let beats = vec![0, 5, 10];
/// let synced = sync(&data, &beats, "mean", true);
/// assert_eq!(synced.shape(), &[2, 2]);
/// ```
pub fn sync(data: &Array2<f32>, idx: &[usize], aggregate: &str, pad: bool) -> Array2<f32> {
    let (n_features, n_frames) = (data.shape()[0], data.shape()[1]);

    if n_features == 0 || n_frames == 0 || idx.is_empty() {
        return Array2::zeros((n_features, 0));
    }

    // Build slice boundaries
    let mut boundaries: Vec<usize> = idx.to_vec();
    boundaries.sort();
    boundaries.dedup();

    if pad {
        if boundaries.is_empty() || boundaries[0] != 0 {
            boundaries.insert(0, 0);
        }
        if *boundaries.last().unwrap() != n_frames {
            boundaries.push(n_frames);
        }
    }

    // Ensure we have at least 2 boundaries to form segments
    if boundaries.len() < 2 {
        return Array2::zeros((n_features, 0));
    }

    let n_segments = boundaries.len() - 1;
    let mut result = Array2::<f32>::zeros((n_features, n_segments));

    for seg in 0..n_segments {
        let start = boundaries[seg];
        let end = boundaries[seg + 1].min(n_frames);

        if start >= end {
            continue;
        }

        for f in 0..n_features {
            let segment_data: Vec<f32> = (start..end).map(|t| data[(f, t)]).collect();

            result[(f, seg)] = match aggregate {
                "mean" => {
                    let sum: f32 = segment_data.iter().sum();
                    sum / segment_data.len() as f32
                }
                "median" => {
                    let mut sorted = segment_data.clone();
                    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    let mid = sorted.len() / 2;
                    if sorted.len().is_multiple_of(2) && sorted.len() > 1 {
                        (sorted[mid - 1] + sorted[mid]) / 2.0
                    } else {
                        sorted[mid]
                    }
                }
                "min" => segment_data.iter().cloned().fold(f32::INFINITY, f32::min),
                "max" => segment_data
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max),
                _ => {
                    // Default to mean
                    let sum: f32 = segment_data.iter().sum();
                    sum / segment_data.len() as f32
                }
            };
        }
    }

    result
}

/// Pick peaks from a signal with threshold and spacing constraints.
pub fn peak_pick(
    x: &[f32],
    pre_max: usize,
    post_max: usize,
    pre_avg: usize,
    post_avg: usize,
    delta: f32,
    wait: usize,
) -> Vec<usize> {
    let mut peaks = Vec::new();
    if x.is_empty() {
        return peaks;
    }

    let mut last_peak = 0usize;
    for i in 0..x.len() {
        if wait > 0 && i < last_peak + wait {
            continue;
        }

        let start_max = i.saturating_sub(pre_max);
        let end_max = (i + post_max + 1).min(x.len());
        let mut is_max = true;
        for j in start_max..end_max {
            if j != i && x[j] >= x[i] {
                is_max = false;
                break;
            }
        }

        if !is_max {
            continue;
        }

        let start_avg = i.saturating_sub(pre_avg);
        let end_avg = (i + post_avg + 1).min(x.len());
        let mut sum = 0.0f32;
        let mut count = 0usize;
        for (j, &val) in x.iter().enumerate().take(end_avg).skip(start_avg) {
            if j != i {
                sum += val;
                count += 1;
            }
        }
        let avg = if count > 0 { sum / count as f32 } else { 0.0 };

        if x[i] >= avg + delta {
            peaks.push(i);
            last_peak = i;
        }
    }

    peaks
}

#[derive(Debug, Clone, Copy)]
pub enum NormType {
    L1,
    L2,
    Max,
}
