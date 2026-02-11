use super::tiny;
use ndarray::Array2;

/// Viterbi decoding for finding the most likely state sequence.
///
/// Uses dynamic programming to find the maximum probability path through
/// a sequence of observations given observation probabilities and transition
/// probabilities.
///
/// # Arguments
/// * `prob` - Observation probability matrix (n_states x n_frames)
/// * `transition` - State transition matrix (n_states x n_states), or None for uniform
/// * `p_init` - Initial state probabilities (n_states), or None for uniform
///
/// # Returns
/// Vector of most likely state indices (length n_frames)
///
/// # Example
/// ```
/// use giggle::utils::viterbi;
/// use ndarray::Array2;
///
/// let prob = Array2::from_shape_vec((2, 4), vec![
///     0.9, 0.8, 0.2, 0.1,  // State 0 probabilities
///     0.1, 0.2, 0.8, 0.9,  // State 1 probabilities
/// ]).unwrap();
///
/// let states = viterbi(&prob, None, None);
/// assert_eq!(states.len(), 4);
/// ```
pub fn viterbi(
    prob: &Array2<f32>,
    transition: Option<&Array2<f32>>,
    p_init: Option<&[f32]>,
) -> Vec<usize> {
    let n_states = prob.shape()[0];
    let n_frames = prob.shape()[1];

    if n_states == 0 || n_frames == 0 {
        return Vec::new();
    }

    // Initialize transition matrix (uniform if not provided)
    let trans = if let Some(t) = transition {
        if t.shape()[0] != n_states || t.shape()[1] != n_states {
            // Invalid transition matrix
            return vec![0; n_frames];
        }
        t.clone()
    } else {
        // Uniform transition
        Array2::from_elem((n_states, n_states), 1.0 / n_states as f32)
    };

    // Initialize state probabilities
    let p_init_vec = if let Some(p) = p_init {
        if p.len() != n_states {
            vec![1.0 / n_states as f32; n_states]
        } else {
            p.to_vec()
        }
    } else {
        vec![1.0 / n_states as f32; n_states]
    };

    // Dynamic programming tables (use log probabilities to avoid underflow)
    let mut delta = Array2::<f32>::zeros((n_states, n_frames));
    let mut psi = Array2::<usize>::zeros((n_states, n_frames));

    // Initialize first frame
    for s in 0..n_states {
        let p_init_log = if p_init_vec[s] > 0.0 {
            p_init_vec[s].ln()
        } else {
            f32::NEG_INFINITY
        };
        let prob_log = if prob[(s, 0)] > 0.0 {
            prob[(s, 0)].ln()
        } else {
            f32::NEG_INFINITY
        };
        delta[(s, 0)] = p_init_log + prob_log;
    }

    // Forward pass
    for t in 1..n_frames {
        for s in 0..n_states {
            let mut max_prob = f32::NEG_INFINITY;
            let mut max_state = 0;

            for prev_s in 0..n_states {
                let trans_log = if trans[(prev_s, s)] > 0.0 {
                    trans[(prev_s, s)].ln()
                } else {
                    f32::NEG_INFINITY
                };
                let prob_val = delta[(prev_s, t - 1)] + trans_log;

                if prob_val > max_prob {
                    max_prob = prob_val;
                    max_state = prev_s;
                }
            }

            let obs_log = if prob[(s, t)] > 0.0 {
                prob[(s, t)].ln()
            } else {
                f32::NEG_INFINITY
            };
            delta[(s, t)] = max_prob + obs_log;
            psi[(s, t)] = max_state;
        }
    }

    // Backtrack to find best path
    let mut states = vec![0; n_frames];

    // Find best final state
    let mut max_prob = f32::NEG_INFINITY;
    let mut best_state = 0;
    for s in 0..n_states {
        if delta[(s, n_frames - 1)] > max_prob {
            max_prob = delta[(s, n_frames - 1)];
            best_state = s;
        }
    }

    states[n_frames - 1] = best_state;

    // Backtrack
    for t in (1..n_frames).rev() {
        states[t - 1] = psi[(states[t], t)];
    }

    states
}

/// Viterbi decoding from discriminative state predictions.
///
/// Given conditional state predictions `prob[s, t]` (probability of state `s`
/// given observation at time `t`), and a transition matrix, computes the most
/// likely state sequence.
///
/// This differs from `viterbi` in that it assumes `prob` contains discriminative
/// probabilities (each column sums to 1) rather than observation likelihoods.
/// The conversion uses Bayes' rule: `P[X | y] ∝ P[Y | x] / P[Y]`
///
/// # Arguments
/// * `prob` - Discriminative state probabilities (n_states x n_steps), each column sums to 1
/// * `transition` - Transition matrix (n_states x n_states), each row sums to 1
/// * `p_state` - Marginal probability distribution over states (default: uniform)
/// * `p_init` - Initial state distribution (default: uniform)
///
/// # Returns
/// Tuple of (most likely state sequence, log probability)
///
/// # Example
/// ```
/// use giggle::utils::viterbi_discriminative;
/// use ndarray::Array2;
///
/// // Discriminative probabilities: each column sums to 1
/// let prob = Array2::from_shape_vec((2, 5), vec![
///     0.8, 0.6, 0.3, 0.4, 0.7,  // P(state=0 | obs)
///     0.2, 0.4, 0.7, 0.6, 0.3,  // P(state=1 | obs)
/// ]).unwrap();
///
/// // Transition matrix: stay in same state with high probability
/// let trans = Array2::from_shape_vec((2, 2), vec![
///     0.9, 0.1,
///     0.1, 0.9,
/// ]).unwrap();
///
/// let (states, logp) = viterbi_discriminative(&prob, &trans, None, None);
/// assert_eq!(states.len(), 5);
/// ```
pub fn viterbi_discriminative(
    prob: &Array2<f32>,
    transition: &Array2<f32>,
    p_state: Option<&[f32]>,
    p_init: Option<&[f32]>,
) -> (Vec<usize>, f32) {
    let n_states = prob.shape()[0];
    let n_steps = prob.shape()[1];

    if n_states == 0 || n_steps == 0 {
        return (Vec::new(), f32::NEG_INFINITY);
    }

    // Validate transition matrix shape
    if transition.shape()[0] != n_states || transition.shape()[1] != n_states {
        return (vec![0; n_steps], f32::NEG_INFINITY);
    }

    let epsilon = tiny(&prob);

    // Set up marginal state distribution (default: uniform)
    let p_state_vec: Vec<f32> = if let Some(p) = p_state {
        if p.len() == n_states {
            p.to_vec()
        } else {
            vec![1.0 / n_states as f32; n_states]
        }
    } else {
        vec![1.0 / n_states as f32; n_states]
    };

    // Set up initial state distribution (default: uniform)
    let p_init_vec: Vec<f32> = if let Some(p) = p_init {
        if p.len() == n_states {
            p.to_vec()
        } else {
            vec![1.0 / n_states as f32; n_states]
        }
    } else {
        vec![1.0 / n_states as f32; n_states]
    };

    // Convert discriminative probabilities to log observation likelihoods
    // By Bayes' rule: log P[X | y] ∝ log P[Y | x] - log P[Y]
    let mut log_prob = Array2::<f32>::zeros((n_states, n_steps));
    for s in 0..n_states {
        let log_marginal = (p_state_vec[s] + epsilon).ln();
        for t in 0..n_steps {
            log_prob[(s, t)] = (prob[(s, t)] + epsilon).ln() - log_marginal;
        }
    }

    // Compute log transition matrix and log initial distribution
    let mut log_trans = Array2::<f32>::zeros((n_states, n_states));
    for i in 0..n_states {
        for j in 0..n_states {
            log_trans[(i, j)] = (transition[(i, j)] + epsilon).ln();
        }
    }

    let log_p_init: Vec<f32> = p_init_vec.iter().map(|&p| (p + epsilon).ln()).collect();

    // Dynamic programming tables
    let mut delta = Array2::<f32>::zeros((n_states, n_steps));
    let mut psi = Array2::<usize>::zeros((n_states, n_steps));

    // Initialize first frame
    for s in 0..n_states {
        delta[(s, 0)] = log_p_init[s] + log_prob[(s, 0)];
    }

    // Forward pass
    for t in 1..n_steps {
        for s in 0..n_states {
            let mut max_prob = f32::NEG_INFINITY;
            let mut max_state = 0;

            for prev_s in 0..n_states {
                let prob_val = delta[(prev_s, t - 1)] + log_trans[(prev_s, s)];
                if prob_val > max_prob {
                    max_prob = prob_val;
                    max_state = prev_s;
                }
            }

            delta[(s, t)] = max_prob + log_prob[(s, t)];
            psi[(s, t)] = max_state;
        }
    }

    // Backtrack to find best path
    let mut states = vec![0; n_steps];
    let mut max_prob = f32::NEG_INFINITY;
    let mut best_state = 0;

    for s in 0..n_states {
        if delta[(s, n_steps - 1)] > max_prob {
            max_prob = delta[(s, n_steps - 1)];
            best_state = s;
        }
    }

    states[n_steps - 1] = best_state;

    for t in (1..n_steps).rev() {
        states[t - 1] = psi[(states[t], t)];
    }

    (states, max_prob)
}

/// Viterbi decoding for binary (multi-label), discriminative state predictions.
///
/// Given binary state predictions `prob[s, t]` indicating the probability of
/// state `s` being active at time `t`, decodes each state independently using
/// a binary Viterbi algorithm.
///
/// Unlike `viterbi_discriminative`, states are not mutually exclusive - multiple
/// states can be active simultaneously.
///
/// # Arguments
/// * `prob` - State probabilities (n_states x n_steps), each value in [0, 1]
/// * `transition` - Either a 2x2 transition matrix (applied to all states) or
///   a (n_states, 2, 2) array where `transition[s]` is the 2x2 matrix for state s
/// * `p_state` - Marginal probability for each state being active (default: 0.5)
/// * `p_init` - Initial probability for each state being active (default: 0.5)
///
/// # Returns
/// Tuple of (binary state matrix, log probabilities per state)
///
/// # Example
/// ```
/// use giggle::utils::viterbi_binary;
/// use ndarray::Array2;
///
/// // Binary state probabilities
/// let prob = Array2::from_shape_vec((1, 10), vec![
///     0.1, 0.7, 0.4, 0.3, 0.8, 0.9, 0.8, 0.2, 0.6, 0.3,
/// ]).unwrap();
///
/// // 2x2 transition matrix: [inactive->inactive, inactive->active; active->inactive, active->active]
/// let trans = Array2::from_shape_vec((2, 2), vec![
///     0.9, 0.1,  // from inactive: 90% stay, 10% activate
///     0.3, 0.7,  // from active: 30% deactivate, 70% stay
/// ]).unwrap();
///
/// let (states, logp) = viterbi_binary(&prob, &trans, None, None);
/// assert_eq!(states.shape(), &[1, 10]);
/// ```
pub fn viterbi_binary(
    prob: &Array2<f32>,
    transition: &Array2<f32>,
    p_state: Option<&[f32]>,
    p_init: Option<&[f32]>,
) -> (Array2<usize>, Vec<f32>) {
    let n_states = prob.shape()[0];
    let n_steps = prob.shape()[1];

    if n_states == 0 || n_steps == 0 {
        return (Array2::zeros((0, 0)), Vec::new());
    }

    // Validate transition matrix (should be 2x2)
    if transition.shape() != [2, 2] {
        // For now, only support shared 2x2 transition matrix
        return (
            Array2::zeros((n_states, n_steps)),
            vec![f32::NEG_INFINITY; n_states],
        );
    }

    // Default marginal state probabilities (0.5 for each)
    let p_state_vec: Vec<f32> = if let Some(p) = p_state {
        if p.len() == n_states {
            p.to_vec()
        } else {
            vec![0.5; n_states]
        }
    } else {
        vec![0.5; n_states]
    };

    // Default initial state probabilities (0.5 for each)
    let p_init_vec: Vec<f32> = if let Some(p) = p_init {
        if p.len() == n_states {
            p.to_vec()
        } else {
            vec![0.5; n_states]
        }
    } else {
        vec![0.5; n_states]
    };

    let mut states = Array2::<usize>::zeros((n_states, n_steps));
    let mut logp = vec![0.0f32; n_states];

    // Process each state independently as a binary (2-state) problem
    for s in 0..n_states {
        // Build binary probability matrix: [P(inactive), P(active)]
        let mut prob_binary = Array2::<f32>::zeros((2, n_steps));
        for t in 0..n_steps {
            prob_binary[(0, t)] = 1.0 - prob[(s, t)]; // P(inactive | obs)
            prob_binary[(1, t)] = prob[(s, t)]; // P(active | obs)
        }

        // Binary state distribution
        let p_state_binary = [1.0 - p_state_vec[s], p_state_vec[s]];
        let p_init_binary = [1.0 - p_init_vec[s], p_init_vec[s]];

        // Run discriminative Viterbi on the binary problem
        let (binary_states, binary_logp) = viterbi_discriminative(
            &prob_binary,
            transition,
            Some(&p_state_binary),
            Some(&p_init_binary),
        );

        // Copy results
        for t in 0..n_steps {
            states[(s, t)] = binary_states[t];
        }
        logp[s] = binary_logp;
    }

    (states, logp)
}

/// Create uniform transition matrix.
///
/// All transitions are equally likely.
pub fn transition_uniform(n_states: usize) -> Array2<f32> {
    Array2::from_elem((n_states, n_states), 1.0 / n_states as f32)
}

/// Create loop transition matrix.
///
/// Favors staying in the same state with probability `p`.
pub fn transition_loop(n_states: usize, p: f32) -> Array2<f32> {
    let mut trans = Array2::zeros((n_states, n_states));
    let p_other = (1.0 - p) / (n_states - 1).max(1) as f32;

    for i in 0..n_states {
        for j in 0..n_states {
            if i == j {
                trans[(i, j)] = p;
            } else {
                trans[(i, j)] = p_other;
            }
        }
    }

    trans
}

/// Create local transition matrix.
///
/// Only allows transitions to adjacent states (i, i-1, i+1).
pub fn transition_local(n_states: usize, p: f32) -> Array2<f32> {
    let mut trans = Array2::zeros((n_states, n_states));
    let p_stay = p;
    let p_move = (1.0 - p) / 2.0;

    for i in 0..n_states {
        trans[(i, i)] = p_stay;
        if i > 0 {
            trans[(i, i - 1)] = p_move;
        }
        if i + 1 < n_states {
            trans[(i, i + 1)] = p_move;
        }
    }

    // Normalize boundary states
    trans[(0, 0)] = p_stay + p_move;
    trans[(n_states - 1, n_states - 1)] = p_stay + p_move;

    trans
}

/// Create cycle transition matrix.
///
/// Favors forward transitions in a cycle: i -> (i+1) mod n_states.
pub fn transition_cycle(n_states: usize, p: f32) -> Array2<f32> {
    let mut trans = Array2::zeros((n_states, n_states));
    let p_forward = p;
    let p_other = (1.0 - p) / (n_states - 1).max(1) as f32;

    for i in 0..n_states {
        for j in 0..n_states {
            if j == (i + 1) % n_states {
                trans[(i, j)] = p_forward;
            } else {
                trans[(i, j)] = p_other;
            }
        }
    }

    trans
}
