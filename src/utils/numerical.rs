use ndarray::Array2;

/// Non-Negative Least Squares (NNLS) solver.
///
/// Solves the constrained optimization problem:
///     minimize ||Ax - b||^2  subject to x >= 0
///
/// Uses the Lawson-Hanson active set algorithm.
///
/// # Arguments
/// * `a` - Matrix A (m x n) as Array2
/// * `b` - Vector b (length m)
/// * `max_iter` - Maximum iterations (default: 3 * n)
///
/// # Returns
/// Solution vector x (length n) where x >= 0 and ||Ax - b||^2 is minimized
///
/// # Example
/// ```
/// use giggle::utils::nnls;
/// use ndarray::Array2;
///
/// let a = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
/// let b = vec![1.0, 2.0, 3.0];
/// let x = nnls(&a, &b, None);
/// assert_eq!(x.len(), 2);
/// assert!(x.iter().all(|&v| v >= 0.0));
/// ```
pub fn nnls(a: &Array2<f32>, b: &[f32], max_iter: Option<usize>) -> Vec<f32> {
    let shape = a.shape();
    let m = shape[0];
    let n = shape[1];

    if m == 0 || n == 0 || b.len() != m {
        return vec![0.0; n];
    }

    let max_iter = max_iter.unwrap_or(3 * n);

    // Initialize solution
    let mut x = vec![0.0f32; n];
    let mut passive_set = vec![false; n]; // false = active (at zero), true = passive (free)

    for _iter in 0..max_iter {
        // Compute residual: r = b - Ax
        let mut residual = b.to_vec();
        for i in 0..m {
            let mut sum = 0.0;
            for j in 0..n {
                sum += a[(i, j)] * x[j];
            }
            residual[i] -= sum;
        }

        // Compute gradient: w = A^T * r
        let mut w = vec![0.0f32; n];
        for j in 0..n {
            let mut sum = 0.0;
            for i in 0..m {
                sum += a[(i, j)] * residual[i];
            }
            w[j] = sum;
        }

        // Check convergence: if all constraints satisfied or gradient is zero for active set
        let mut max_w = 0.0f32;
        let mut max_idx = 0;
        for j in 0..n {
            if !passive_set[j] && w[j] > max_w {
                max_w = w[j];
                max_idx = j;
            }
        }

        if max_w <= 1e-10 {
            break; // Converged
        }

        // Add variable to passive set
        passive_set[max_idx] = true;

        // Solve unconstrained least squares for passive set
        loop {
            // Construct reduced problem for passive variables
            let passive_indices: Vec<usize> = (0..n).filter(|&i| passive_set[i]).collect();
            if passive_indices.is_empty() {
                break;
            }

            // Solve A_p * x_p = b using normal equations
            let n_passive = passive_indices.len();
            let mut ata = vec![0.0f32; n_passive * n_passive];
            let mut atb = vec![0.0f32; n_passive];

            for (i, &pi) in passive_indices.iter().enumerate() {
                for (j, &pj) in passive_indices.iter().enumerate() {
                    let mut sum = 0.0;
                    for k in 0..m {
                        sum += a[(k, pi)] * a[(k, pj)];
                    }
                    ata[i * n_passive + j] = sum;
                }

                let mut sum = 0.0;
                for k in 0..m {
                    sum += a[(k, pi)] * b[k];
                }
                atb[i] = sum;
            }

            // Solve using simple Gauss elimination (for small systems)
            let x_passive = solve_linear_system(&ata, &atb, n_passive);

            // Check if solution is feasible (all positive)
            let mut min_val = f32::INFINITY;
            let mut min_idx = 0;
            for (i, &_pi) in passive_indices.iter().enumerate() {
                if x_passive[i] < min_val {
                    min_val = x_passive[i];
                    min_idx = i;
                }
            }

            if min_val >= -1e-10 {
                // Solution is feasible, update x
                for (i, &pi) in passive_indices.iter().enumerate() {
                    x[pi] = x_passive[i].max(0.0);
                }
                break;
            } else {
                // Infeasible, remove most negative variable from passive set
                let remove_idx = passive_indices[min_idx];
                passive_set[remove_idx] = false;
                x[remove_idx] = 0.0;
            }
        }
    }

    x
}

fn solve_linear_system(a: &[f32], b: &[f32], n: usize) -> Vec<f32> {
    if n == 0 {
        return Vec::new();
    }

    let mut aug = vec![0.0f32; n * (n + 1)];
    for i in 0..n {
        for j in 0..n {
            aug[i * (n + 1) + j] = a[i * n + j];
        }
        aug[i * (n + 1) + n] = b[i];
    }

    // Forward elimination
    for i in 0..n {
        // Find pivot
        let mut max_row = i;
        for k in (i + 1)..n {
            if aug[k * (n + 1) + i].abs() > aug[max_row * (n + 1) + i].abs() {
                max_row = k;
            }
        }

        // Swap rows
        if max_row != i {
            for j in 0..=(n) {
                aug.swap(i * (n + 1) + j, max_row * (n + 1) + j);
            }
        }

        // Eliminate column
        let pivot = aug[i * (n + 1) + i];
        if pivot.abs() < 1e-10 {
            continue;
        }

        for k in (i + 1)..n {
            let factor = aug[k * (n + 1) + i] / pivot;
            for j in i..=(n) {
                aug[k * (n + 1) + j] -= factor * aug[i * (n + 1) + j];
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0f32; n];
    for i in (0..n).rev() {
        let mut sum = aug[i * (n + 1) + n];
        for j in (i + 1)..n {
            sum -= aug[i * (n + 1) + j] * x[j];
        }
        let diag = aug[i * (n + 1) + i];
        x[i] = if diag.abs() > 1e-10 { sum / diag } else { 0.0 };
    }

    x
}

/// Non-negative Matrix Factorization (NMF) using multiplicative update rules.
///
/// Decomposes a non-negative matrix V into two non-negative matrices W and H
/// such that V ≈ W * H.
///
/// This uses the multiplicative update rules from Lee and Seung (2001).
///
/// # Arguments
/// * `v` - Input non-negative matrix (n_features x n_samples)
/// * `n_components` - Number of components
/// * `max_iter` - Maximum number of iterations (default: 200)
/// * `tol` - Tolerance for convergence (default: 1e-4)
///
/// # Returns
/// Tuple of (W, H) where:
/// - W: Component matrix (n_features x n_components)
/// - H: Activation matrix (n_components x n_samples)
///
/// # Example
/// ```
/// use giggle::utils::nmf;
/// use ndarray::Array2;
///
/// let v = Array2::from_shape_vec((4, 6), vec![
///     1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
///     2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
///     1.0, 1.0, 2.0, 2.0, 3.0, 3.0,
///     3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
/// ]).unwrap();
/// let (w, h) = nmf(&v, 2, Some(100), None);
/// assert_eq!(w.shape(), &[4, 2]);
/// assert_eq!(h.shape(), &[2, 6]);
/// ```
pub fn nmf(
    v: &Array2<f32>,
    n_components: usize,
    max_iter: Option<usize>,
    tol: Option<f32>,
) -> (Array2<f32>, Array2<f32>) {
    let n_features = v.shape()[0];
    let n_samples = v.shape()[1];

    if n_features == 0 || n_samples == 0 || n_components == 0 {
        return (
            Array2::zeros((n_features, 0)),
            Array2::zeros((0, n_samples)),
        );
    }

    let max_iter = max_iter.unwrap_or(200);
    let tol = tol.unwrap_or(1e-4);
    let eps = 1e-10f32; // Small value to avoid division by zero

    // Initialize W and H with random positive values
    // Use deterministic initialization based on data for reproducibility
    let mut w = Array2::<f32>::zeros((n_features, n_components));
    let mut h = Array2::<f32>::zeros((n_components, n_samples));

    // Initialize W: use columns from V or random if not enough
    let v_mean = v.mean().unwrap_or(1.0).max(0.01);
    for k in 0..n_components {
        for i in 0..n_features {
            // Initialize with scaled value based on position
            let seed_val = ((i * n_components + k + 1) as f32 * 0.1).sin().abs() + 0.1;
            w[(i, k)] = seed_val * v_mean.sqrt();
        }
    }

    // Initialize H similarly
    for k in 0..n_components {
        for j in 0..n_samples {
            let seed_val = ((k * n_samples + j + 1) as f32 * 0.1).cos().abs() + 0.1;
            h[(k, j)] = seed_val * v_mean.sqrt();
        }
    }

    let mut prev_error = f32::MAX;

    for _iter in 0..max_iter {
        // Update H: H <- H .* (W^T * V) ./ (W^T * W * H + eps)
        // Compute W^T * V
        let mut wt_v = Array2::<f32>::zeros((n_components, n_samples));
        for k in 0..n_components {
            for j in 0..n_samples {
                let mut sum = 0.0;
                for i in 0..n_features {
                    sum += w[(i, k)] * v[(i, j)];
                }
                wt_v[(k, j)] = sum;
            }
        }

        // Compute W^T * W
        let mut wt_w = Array2::<f32>::zeros((n_components, n_components));
        for k1 in 0..n_components {
            for k2 in 0..n_components {
                let mut sum = 0.0;
                for i in 0..n_features {
                    sum += w[(i, k1)] * w[(i, k2)];
                }
                wt_w[(k1, k2)] = sum;
            }
        }

        // Compute W^T * W * H
        let mut wt_w_h = Array2::<f32>::zeros((n_components, n_samples));
        for k in 0..n_components {
            for j in 0..n_samples {
                let mut sum = 0.0;
                for k2 in 0..n_components {
                    sum += wt_w[(k, k2)] * h[(k2, j)];
                }
                wt_w_h[(k, j)] = sum;
            }
        }

        // Update H
        for k in 0..n_components {
            for j in 0..n_samples {
                h[(k, j)] *= wt_v[(k, j)] / (wt_w_h[(k, j)] + eps);
            }
        }

        // Update W: W <- W .* (V * H^T) ./ (W * H * H^T + eps)
        // Compute V * H^T
        let mut v_ht = Array2::<f32>::zeros((n_features, n_components));
        for i in 0..n_features {
            for k in 0..n_components {
                let mut sum = 0.0;
                for j in 0..n_samples {
                    sum += v[(i, j)] * h[(k, j)];
                }
                v_ht[(i, k)] = sum;
            }
        }

        // Compute H * H^T
        let mut h_ht = Array2::<f32>::zeros((n_components, n_components));
        for k1 in 0..n_components {
            for k2 in 0..n_components {
                let mut sum = 0.0;
                for j in 0..n_samples {
                    sum += h[(k1, j)] * h[(k2, j)];
                }
                h_ht[(k1, k2)] = sum;
            }
        }

        // Compute W * H * H^T
        let mut w_h_ht = Array2::<f32>::zeros((n_features, n_components));
        for i in 0..n_features {
            for k in 0..n_components {
                let mut sum = 0.0;
                for k2 in 0..n_components {
                    sum += w[(i, k2)] * h_ht[(k2, k)];
                }
                w_h_ht[(i, k)] = sum;
            }
        }

        // Update W
        for i in 0..n_features {
            for k in 0..n_components {
                w[(i, k)] *= v_ht[(i, k)] / (w_h_ht[(i, k)] + eps);
            }
        }

        // Check convergence: compute reconstruction error
        let mut error = 0.0f32;
        for i in 0..n_features {
            for j in 0..n_samples {
                let mut approx = 0.0;
                for k in 0..n_components {
                    approx += w[(i, k)] * h[(k, j)];
                }
                let diff = v[(i, j)] - approx;
                error += diff * diff;
            }
        }
        error = error.sqrt();

        if (prev_error - error).abs() < tol {
            break;
        }
        prev_error = error;
    }

    (w, h)
}

/// Decompose a spectrogram using NMF.
///
/// Given a spectrogram S, produces a decomposition into components and activations
/// such that S ≈ components * activations.
///
/// # Arguments
/// * `s` - Input spectrogram (n_features x n_samples), should be non-negative
/// * `n_components` - Number of components (default: n_features)
/// * `max_iter` - Maximum iterations for NMF (default: 200)
/// * `sort` - If true, sort components by ascending peak frequency
///
/// # Returns
/// Tuple of (components, activations) where:
/// - components: Matrix of basis elements (n_features x n_components)
/// - activations: Activation/coefficient matrix (n_components x n_samples)
///
/// # Example
/// ```
/// use giggle::utils::decompose;
/// use ndarray::Array2;
///
/// let spec = Array2::from_shape_vec((4, 8), vec![
///     1.0, 2.0, 1.5, 2.5, 1.0, 2.0, 1.5, 2.5,
///     2.0, 3.0, 2.5, 3.5, 2.0, 3.0, 2.5, 3.5,
///     0.5, 1.0, 0.8, 1.2, 0.5, 1.0, 0.8, 1.2,
///     1.5, 2.5, 2.0, 3.0, 1.5, 2.5, 2.0, 3.0,
/// ]).unwrap();
/// let (comps, acts) = decompose(&spec, Some(2), None, false);
/// assert_eq!(comps.shape(), &[4, 2]);
/// assert_eq!(acts.shape(), &[2, 8]);
/// ```
pub fn decompose(
    s: &Array2<f32>,
    n_components: Option<usize>,
    max_iter: Option<usize>,
    sort: bool,
) -> (Array2<f32>, Array2<f32>) {
    let n_features = s.shape()[0];
    let n_samples = s.shape()[1];

    if n_features == 0 || n_samples == 0 {
        return (Array2::zeros((0, 0)), Array2::zeros((0, 0)));
    }

    let n_components = n_components.unwrap_or(n_features);

    // Ensure non-negative input
    let mut s_nn = s.clone();
    for val in s_nn.iter_mut() {
        if *val < 0.0 {
            *val = 0.0;
        }
    }

    // Perform NMF
    let (mut components, mut activations) = nmf(&s_nn, n_components, max_iter, None);

    // Sort by peak frequency if requested
    if sort && n_components > 0 {
        // Find peak index for each component
        let mut peak_indices: Vec<(usize, usize)> = (0..n_components)
            .map(|k| {
                let mut max_val = 0.0f32;
                let mut max_idx = 0;
                for i in 0..n_features {
                    if components[(i, k)] > max_val {
                        max_val = components[(i, k)];
                        max_idx = i;
                    }
                }
                (k, max_idx)
            })
            .collect();

        // Sort by peak index
        peak_indices.sort_by_key(|&(_, idx)| idx);

        // Reorder components and activations
        let mut sorted_components = Array2::<f32>::zeros((n_features, n_components));
        let mut sorted_activations = Array2::<f32>::zeros((n_components, n_samples));

        for (new_k, &(old_k, _)) in peak_indices.iter().enumerate() {
            for i in 0..n_features {
                sorted_components[(i, new_k)] = components[(i, old_k)];
            }
            for j in 0..n_samples {
                sorted_activations[(new_k, j)] = activations[(old_k, j)];
            }
        }

        components = sorted_components;
        activations = sorted_activations;
    }

    (components, activations)
}

/// Harmonic-Percussive Source Separation using NMF.
///
/// Separates a spectrogram into harmonic and percussive components using
/// Non-negative Matrix Factorization with sparseness constraints.
///
/// This is different from the median-filtering HPSS in the effects module.
/// The NMF-based approach learns basis functions that represent harmonic
/// and percussive patterns.
///
/// # Arguments
/// * `s` - Input magnitude spectrogram (n_freq x n_frames)
/// * `n_harmonic` - Number of harmonic components (default: n_freq/4)
/// * `n_percussive` - Number of percussive components (default: n_freq/4)
/// * `max_iter` - Maximum NMF iterations (default: 100)
///
/// # Returns
/// Tuple of (harmonic, percussive) spectrograms with same shape as input
///
/// # Example
/// ```
/// use giggle::utils::nmf_hpss;
/// use ndarray::Array2;
///
/// let spec = Array2::from_shape_vec((8, 10), (0..80).map(|i| (i as f32 * 0.1).sin().abs() + 0.1).collect()).unwrap();
/// let (harmonic, percussive) = nmf_hpss(&spec, None, None, None);
/// assert_eq!(harmonic.shape(), spec.shape());
/// assert_eq!(percussive.shape(), spec.shape());
/// ```
pub fn nmf_hpss(
    s: &Array2<f32>,
    n_harmonic: Option<usize>,
    n_percussive: Option<usize>,
    max_iter: Option<usize>,
) -> (Array2<f32>, Array2<f32>) {
    let n_freq = s.shape()[0];
    let n_frames = s.shape()[1];

    if n_freq == 0 || n_frames == 0 {
        return (Array2::zeros((0, 0)), Array2::zeros((0, 0)));
    }

    let n_harmonic = n_harmonic.unwrap_or((n_freq / 4).max(1));
    let n_percussive = n_percussive.unwrap_or((n_freq / 4).max(1));
    let n_components = n_harmonic + n_percussive;
    let max_iter = max_iter.unwrap_or(100);

    // Perform NMF decomposition
    let (components, activations) = nmf(s, n_components, Some(max_iter), None);

    // Classify components as harmonic or percussive based on their temporal smoothness
    // Harmonic components have smooth temporal activations
    // Percussive components have sparse/impulsive temporal activations
    let mut component_scores: Vec<(usize, f32)> = (0..n_components)
        .map(|k| {
            // Compute temporal variance of activation (lower = more harmonic)
            let mean: f32 =
                (0..n_frames).map(|j| activations[(k, j)]).sum::<f32>() / n_frames as f32;
            let variance: f32 = (0..n_frames)
                .map(|j| {
                    let diff = activations[(k, j)] - mean;
                    diff * diff
                })
                .sum::<f32>()
                / n_frames as f32;

            // Also consider spectral smoothness (harmonic = smoother spectrum)
            let spectral_diff: f32 = (1..n_freq)
                .map(|i| (components[(i, k)] - components[(i - 1, k)]).abs())
                .sum::<f32>();

            // Combined score: higher = more percussive
            let score = variance * spectral_diff;
            (k, score)
        })
        .collect();

    // Sort by score (ascending = harmonic first)
    component_scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    // Split into harmonic and percussive components
    let harmonic_indices: Vec<usize> = component_scores[..n_harmonic.min(n_components)]
        .iter()
        .map(|&(k, _)| k)
        .collect();
    let percussive_indices: Vec<usize> = component_scores[n_harmonic.min(n_components)..]
        .iter()
        .map(|&(k, _)| k)
        .collect();

    // Reconstruct harmonic and percussive spectrograms
    let mut harmonic = Array2::<f32>::zeros((n_freq, n_frames));
    let mut percussive = Array2::<f32>::zeros((n_freq, n_frames));

    for i in 0..n_freq {
        for j in 0..n_frames {
            for &k in &harmonic_indices {
                harmonic[(i, j)] += components[(i, k)] * activations[(k, j)];
            }
            for &k in &percussive_indices {
                percussive[(i, j)] += components[(i, k)] * activations[(k, j)];
            }
        }
    }

    (harmonic, percussive)
}
