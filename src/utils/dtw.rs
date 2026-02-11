use ndarray::Array2;

/// Compute Dynamic Time Warping (DTW) distance and path.
///
/// DTW finds the optimal alignment between two time series by warping
/// the time axis. Useful for music synchronization and pattern matching.
///
/// # Arguments
/// * `x` - First feature matrix (n_features x n_frames_x)
/// * `y` - Second feature matrix (n_features x n_frames_y)
/// * `metric` - Distance metric: "euclidean", "manhattan", or "cosine"
///
/// # Returns
/// Tuple of (distance, path) where path is Vec<(i, j)> of aligned frame indices
///
/// # Example
/// ```
/// use giggle::utils::dtw;
/// use ndarray::Array2;
///
/// let x = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]).unwrap();
/// let y = Array2::from_shape_vec((2, 4), vec![1.0, 2.0, 2.5, 3.0, 1.0, 2.0, 2.5, 3.0]).unwrap();
/// let (distance, path) = dtw(&x, &y, "euclidean");
/// assert!(distance >= 0.0);
/// assert!(path.len() > 0);
/// ```
pub fn dtw(x: &Array2<f32>, y: &Array2<f32>, metric: &str) -> (f32, Vec<(usize, usize)>) {
    let n_features_x = x.shape()[0];
    let n_features_y = y.shape()[0];
    let n_x = x.shape()[1];
    let n_y = y.shape()[1];

    if n_features_x != n_features_y || n_x == 0 || n_y == 0 {
        return (f32::INFINITY, Vec::new());
    }

    // Compute pairwise distance matrix
    let mut dist = Array2::<f32>::zeros((n_x, n_y));
    for i in 0..n_x {
        for j in 0..n_y {
            let mut d = 0.0f64;
            match metric {
                "euclidean" => {
                    for k in 0..n_features_x {
                        let diff = x[(k, i)] as f64 - y[(k, j)] as f64;
                        d += diff * diff;
                    }
                    d = d.sqrt();
                }
                "manhattan" => {
                    for k in 0..n_features_x {
                        d += (x[(k, i)] - y[(k, j)]).abs() as f64;
                    }
                }
                "cosine" => {
                    let mut dot = 0.0f64;
                    let mut norm_x = 0.0f64;
                    let mut norm_y = 0.0f64;
                    for k in 0..n_features_x {
                        let vx = x[(k, i)] as f64;
                        let vy = y[(k, j)] as f64;
                        dot += vx * vy;
                        norm_x += vx * vx;
                        norm_y += vy * vy;
                    }
                    let norm_prod = (norm_x * norm_y).sqrt();
                    if norm_prod > 1e-10 {
                        d = 1.0 - (dot / norm_prod); // Cosine distance
                    } else {
                        d = 1.0;
                    }
                }
                _ => {
                    return (f32::INFINITY, Vec::new());
                }
            }
            dist[(i, j)] = d as f32;
        }
    }

    // Compute accumulated cost matrix
    let mut cost = Array2::<f32>::zeros((n_x, n_y));
    cost[(0, 0)] = dist[(0, 0)];

    // Initialize first row and column
    for i in 1..n_x {
        cost[(i, 0)] = cost[(i - 1, 0)] + dist[(i, 0)];
    }
    for j in 1..n_y {
        cost[(0, j)] = cost[(0, j - 1)] + dist[(0, j)];
    }

    // Fill cost matrix
    for i in 1..n_x {
        for j in 1..n_y {
            let min_cost = cost[(i - 1, j)]
                .min(cost[(i, j - 1)])
                .min(cost[(i - 1, j - 1)]);
            cost[(i, j)] = dist[(i, j)] + min_cost;
        }
    }

    // Backtrack to find optimal path
    let mut path = Vec::new();
    let mut i = n_x - 1;
    let mut j = n_y - 1;
    path.push((i, j));

    while i > 0 || j > 0 {
        if i == 0 {
            j -= 1;
        } else if j == 0 {
            i -= 1;
        } else {
            // Choose direction with minimum cost
            let diag = cost[(i - 1, j - 1)];
            let left = cost[(i, j - 1)];
            let down = cost[(i - 1, j)];

            if diag <= left && diag <= down {
                i -= 1;
                j -= 1;
            } else if left <= down {
                j -= 1;
            } else {
                i -= 1;
            }
        }
        path.push((i, j));
    }

    path.reverse();
    (cost[(n_x - 1, n_y - 1)], path)
}

/// Compute DTW distance only (without backtracking path).
///
/// Faster than full DTW when only the distance is needed.
///
/// # Arguments
/// * `x` - First feature matrix (n_features x n_frames_x)
/// * `y` - Second feature matrix (n_features x n_frames_y)
/// * `metric` - Distance metric
///
/// # Returns
/// DTW distance
pub fn dtw_distance(x: &Array2<f32>, y: &Array2<f32>, metric: &str) -> f32 {
    let (distance, _) = dtw(x, y, metric);
    distance
}

/// DTW with step matrix recording for external backtracking.
///
/// This variant records the step matrix which can be used with `dtw_backtracking`
/// to reconstruct the warping path with custom step sizes.
///
/// # Arguments
/// * `x` - First feature matrix (n_features x n_frames_x)
/// * `y` - Second feature matrix (n_features x n_frames_y)
/// * `metric` - Distance metric ("euclidean", "manhattan", "cosine")
///
/// # Returns
/// Tuple of (distance, steps_matrix) where steps_matrix contains the index
/// of the step used to reach each cell.
///
/// # Example
/// ```
/// use giggle::utils::dtw_with_steps;
/// use ndarray::Array2;
///
/// let x = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]).unwrap();
/// let y = Array2::from_shape_vec((2, 4), vec![1.0, 2.0, 2.5, 3.0, 1.0, 2.0, 2.5, 3.0]).unwrap();
/// let (distance, steps) = dtw_with_steps(&x, &y, "euclidean");
/// assert!(distance >= 0.0);
/// assert_eq!(steps.shape(), &[3, 4]);
/// ```
pub fn dtw_with_steps(x: &Array2<f32>, y: &Array2<f32>, metric: &str) -> (f32, Array2<usize>) {
    let n_features_x = x.shape()[0];
    let n_features_y = y.shape()[0];
    let n_x = x.shape()[1];
    let n_y = y.shape()[1];

    if n_features_x != n_features_y || n_x == 0 || n_y == 0 {
        return (f32::INFINITY, Array2::zeros((0, 0)));
    }

    // Compute pairwise distance matrix
    let mut dist = Array2::<f32>::zeros((n_x, n_y));
    for i in 0..n_x {
        for j in 0..n_y {
            let mut d = 0.0f64;
            match metric {
                "euclidean" => {
                    for k in 0..n_features_x {
                        let diff = x[(k, i)] as f64 - y[(k, j)] as f64;
                        d += diff * diff;
                    }
                    d = d.sqrt();
                }
                "manhattan" => {
                    for k in 0..n_features_x {
                        d += (x[(k, i)] - y[(k, j)]).abs() as f64;
                    }
                }
                "cosine" => {
                    let mut dot = 0.0f64;
                    let mut norm_x = 0.0f64;
                    let mut norm_y = 0.0f64;
                    for k in 0..n_features_x {
                        let vx = x[(k, i)] as f64;
                        let vy = y[(k, j)] as f64;
                        dot += vx * vy;
                        norm_x += vx * vx;
                        norm_y += vy * vy;
                    }
                    let norm_prod = (norm_x * norm_y).sqrt();
                    if norm_prod > 1e-10 {
                        d = 1.0 - (dot / norm_prod);
                    } else {
                        d = 1.0;
                    }
                }
                _ => {
                    return (f32::INFINITY, Array2::zeros((0, 0)));
                }
            }
            dist[(i, j)] = d as f32;
        }
    }

    // Compute accumulated cost matrix and record steps
    let mut cost = Array2::<f32>::zeros((n_x, n_y));
    let mut steps = Array2::<usize>::zeros((n_x, n_y));

    cost[(0, 0)] = dist[(0, 0)];
    steps[(0, 0)] = 0; // Starting point

    // Initialize first row and column
    for i in 1..n_x {
        cost[(i, 0)] = cost[(i - 1, 0)] + dist[(i, 0)];
        steps[(i, 0)] = 2; // Step from above (1, 0)
    }
    for j in 1..n_y {
        cost[(0, j)] = cost[(0, j - 1)] + dist[(0, j)];
        steps[(0, j)] = 1; // Step from left (0, 1)
    }

    // Fill cost matrix
    // Step indices: 0 = diagonal (1,1), 1 = left (0,1), 2 = up (1,0)
    for i in 1..n_x {
        for j in 1..n_y {
            let diag_cost = cost[(i - 1, j - 1)];
            let left_cost = cost[(i, j - 1)];
            let up_cost = cost[(i - 1, j)];

            // Find minimum cost and record step
            if diag_cost <= left_cost && diag_cost <= up_cost {
                cost[(i, j)] = dist[(i, j)] + diag_cost;
                steps[(i, j)] = 0; // Diagonal
            } else if left_cost <= up_cost {
                cost[(i, j)] = dist[(i, j)] + left_cost;
                steps[(i, j)] = 1; // Left
            } else {
                cost[(i, j)] = dist[(i, j)] + up_cost;
                steps[(i, j)] = 2; // Up
            }
        }
    }

    (cost[(n_x - 1, n_y - 1)], steps)
}

/// Backtrack a DTW warping path from a step matrix.
///
/// Uses the saved step sizes from the cost accumulation step to backtrack
/// the index pairs for a warping path.
///
/// # Arguments
/// * `steps` - Step matrix from `dtw_with_steps`, containing indices of used steps
/// * `step_sizes_sigma` - Optional custom step sizes as &[(di, dj)].
///   If None, uses default steps: [(1, 1), (0, 1), (1, 0)]
/// * `subseq` - Enable subsequence DTW (for retrieval tasks)
/// * `start` - Start column index for backtracking (only used when subseq=true)
///
/// # Returns
/// Vector of (i, j) index pairs representing the warping path
///
/// # Example
/// ```
/// use giggle::utils::{dtw_with_steps, dtw_backtracking};
/// use ndarray::Array2;
///
/// let x = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]).unwrap();
/// let y = Array2::from_shape_vec((2, 4), vec![1.0, 2.0, 2.5, 3.0, 1.0, 2.0, 2.5, 3.0]).unwrap();
/// let (distance, steps) = dtw_with_steps(&x, &y, "euclidean");
/// let path = dtw_backtracking(&steps, None, false, None);
/// assert!(!path.is_empty());
/// ```
pub fn dtw_backtracking(
    steps: &Array2<usize>,
    step_sizes_sigma: Option<&[(usize, usize)]>,
    subseq: bool,
    start: Option<usize>,
) -> Vec<(usize, usize)> {
    let n_x = steps.shape()[0];
    let n_y = steps.shape()[1];

    if n_x == 0 || n_y == 0 {
        return Vec::new();
    }

    // Validate start parameter
    if !subseq && start.is_some() {
        return Vec::new();
    }

    // Default step sizes: diagonal, left, up
    let default_steps = [(1, 1), (0, 1), (1, 0)];
    let step_sizes = step_sizes_sigma.unwrap_or(&default_steps);

    // Combine default and custom steps
    let all_steps: Vec<(usize, usize)> = if step_sizes_sigma.is_some() {
        default_steps
            .iter()
            .chain(step_sizes.iter())
            .copied()
            .collect()
    } else {
        default_steps.to_vec()
    };

    // Determine starting position
    let mut cur_idx = if let Some(s) = start {
        if s >= n_y {
            return Vec::new();
        }
        (n_x - 1, s)
    } else {
        (n_x - 1, n_y - 1)
    };

    let mut path = Vec::new();
    path.push(cur_idx);

    // Backtrack
    // Stop criteria: reach (0, 0) for full DTW, or first row for subsequence
    while (subseq && cur_idx.0 > 0) || (!subseq && cur_idx != (0, 0)) {
        let step_idx = steps[cur_idx];

        if step_idx >= all_steps.len() {
            // Invalid step index, break to avoid panic
            break;
        }

        let (di, dj) = all_steps[step_idx];

        // Check bounds before subtraction
        if cur_idx.0 < di || cur_idx.1 < dj {
            break;
        }

        cur_idx = (cur_idx.0 - di, cur_idx.1 - dj);
        path.push(cur_idx);

        // Safety check: if we reached (0, 0), stop
        if cur_idx == (0, 0) {
            break;
        }
    }

    path.reverse();
    path
}
