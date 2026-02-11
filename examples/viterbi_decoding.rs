//! Viterbi Decoding Example
//!
//! This example demonstrates how to use Viterbi decoding to impose temporal
//! smoothing on frame-wise state predictions.
//!
//! The working example is silence/non-silence detection, showing how Viterbi
//! can reduce spurious state changes caused by brief energy dips.

use giggle::utils;
use log::info;
use ndarray::Array2;

fn main() {
    env_logger::init();
    info!("Viterbi Decoding Example");

    // Example 1: Silence Detection with Frame-wise Predictions
    info!("Example 1: Silence/Non-Silence Detection");

    // Simulate RMS energy values for a signal with periods of silence
    // Format: [silence_start, speech, brief_dip, speech, silence_end]
    let rms_values = vec![
        0.005, 0.005, 0.005, // Silence (below threshold)
        0.15, 0.18, 0.12, 0.20, 0.16,  // Speech
        0.008, // Brief dip (momentary silence)
        0.14, 0.19, 0.17, 0.13, // Speech continues
        0.004, 0.005, 0.006, // Silence at end
    ];

    let threshold = 0.02;
    let times: Vec<f32> = (0..rms_values.len()).map(|i| i as f32 * 0.01).collect();

    info!("Simulated RMS values (threshold = {:.3}):", threshold);
    for (i, (t, rms)) in times.iter().zip(rms_values.iter()).enumerate() {
        let state = if *rms >= threshold {
            "SPEECH"
        } else {
            "SILENCE"
        };
        let marker = if i == 8 { " <-- brief dip" } else { "" };
        info!("  t={:.2}s: RMS={:.3} -> {}{}", t, rms, state, marker);
    }

    // Convert RMS to probabilities using logistic mapping
    let r_normalized: Vec<f32> = rms_values
        .iter()
        .map(|rms| (rms - threshold) / 0.05) // Normalize by std dev estimate
        .collect();

    let p_non_silent: Vec<f32> = r_normalized
        .iter()
        .map(|r| 1.0 / (1.0 + (-r).exp()))
        .collect();

    info!("\nFrame-wise probabilities (P(non-silent | x)):");
    for (i, (t, p)) in times.iter().zip(p_non_silent.iter()).enumerate() {
        let decision = if *p >= 0.5 { "SPEECH" } else { "SILENCE" };
        let marker = if i == 8 { " <-- false transition" } else { "" };
        info!("  t={:.2}s: P={:.3} -> {}{}", t, p, decision, marker);
    }

    // Viterbi Decoding
    info!("Viterbi Decoding");

    // Build probability matrix (n_states x n_frames)
    // State 0 = silent, State 1 = non-silent
    let n_frames = p_non_silent.len();
    let mut prob = Array2::<f32>::zeros((2, n_frames));
    for t in 0..n_frames {
        prob[(0, t)] = 1.0 - p_non_silent[t]; // P(silent | x)
        prob[(1, t)] = p_non_silent[t]; // P(non-silent | x)
    }

    // Create transition matrix with self-loop bias
    // Higher self-loop probability = smoother transitions
    let transition = utils::transition_loop(2, 0.7);
    info!("Transition matrix (self-loop prob = 0.7):");
    info!("  [[{:.2}, {:.2}],", transition[(0, 0)], transition[(0, 1)]);
    info!("   [{:.2}, {:.2}]]", transition[(1, 0)], transition[(1, 1)]);

    // Run Viterbi decoding
    let (states, log_prob) = utils::viterbi_discriminative(
        &prob,
        &transition,
        None, // uniform p_state
        None, // uniform p_init
    );

    info!("\nViterbi result (log prob = {:.4}):", log_prob);
    for (i, (t, state)) in times.iter().zip(states.iter()).enumerate() {
        let state_name = if *state == 0 { "SILENCE" } else { "SPEECH" };
        let frame_decision = if p_non_silent[i] >= 0.5 {
            "SPEECH"
        } else {
            "SILENCE"
        };
        let marker = if *state != (if p_non_silent[i] >= 0.5 { 1 } else { 0 }) {
            " <-- corrected"
        } else {
            ""
        };
        info!(
            "  t={:.2}s: frame={:7}, viterbi={:7}{}",
            t, frame_decision, state_name, marker
        );
    }

    // Example 2: Effect of Different Transition Probabilities
    info!("Example 2: Effect of Transition Smoothing");

    let self_loop_probs = [0.5, 0.7, 0.9, 0.95];

    for p_self in &self_loop_probs {
        let trans = utils::transition_loop(2, *p_self);
        let (states_viterbi, _) = utils::viterbi_discriminative(&prob, &trans, None, None);

        // Count state transitions
        let mut transitions = 0;
        for i in 1..states_viterbi.len() {
            if states_viterbi[i] != states_viterbi[i - 1] {
                transitions += 1;
            }
        }

        info!(
            "  Self-loop prob = {:.2}: {} transitions",
            p_self, transitions
        );
    }

    // Example 3: Binary Viterbi for Multi-State Problems
    info!("Example 3: Binary Viterbi (Multi-label)");

    // Simulate a problem with two independent binary states:
    // - State 0: Speech presence (0=absent, 1=present)
    // - State 1: Music presence (0=absent, 1=present)

    let speech_prob = [0.1, 0.2, 0.8, 0.9, 0.3, 0.4, 0.7, 0.85];
    let music_prob = [0.8, 0.9, 0.2, 0.1, 0.7, 0.8, 0.3, 0.2];

    // Build binary probability matrix (n_states x n_frames)
    let mut binary_prob = Array2::<f32>::zeros((2, speech_prob.len()));
    info!("Multi-label classification (speech + music):");
    for t in 0..speech_prob.len() {
        binary_prob[(0, t)] = speech_prob[t]; // Speech probability
        binary_prob[(1, t)] = music_prob[t]; // Music probability
    }

    // Binary transition matrix: [inactive->inactive, inactive->active; active->inactive, active->active]
    let binary_trans = Array2::from_shape_vec(
        (2, 2),
        vec![
            0.8, 0.2, // from inactive: 80% stay, 20% activate
            0.3, 0.7,
        ], // from active: 30% deactivate, 70% stay
    )
    .unwrap();

    let (binary_states, log_probs) = utils::viterbi_binary(&binary_prob, &binary_trans, None, None);

    for t in 0..speech_prob.len() {
        let speech_state = if binary_states[(0, t)] == 1 {
            "PRESENT"
        } else {
            "ABSENT "
        };
        let music_state = if binary_states[(1, t)] == 1 {
            "PRESENT"
        } else {
            "ABSENT "
        };
        info!(
            "  {:5} | {:11.2} | {:10.2} | {:12} | {}",
            t, speech_prob[t], music_prob[t], speech_state, music_state
        );
    }
    info!(
        "\n  Log probabilities per state: [{:.2}, {:.2}]",
        log_probs[0], log_probs[1]
    );

    // Example 4: Different Transition Types
    info!("Example 4: Different Transition Types");

    // Create a sequence with 3 states (e.g., silence, speech, music)
    let n_states = 3;

    info!("Transition matrix types for {} states:", n_states);

    // Uniform
    info!("\n  Uniform (all transitions equal):");
    let uniform = utils::transition_uniform(n_states);
    for i in 0..n_states {
        let row: Vec<String> = (0..n_states)
            .map(|j| format!("{:.3}", uniform[(i, j)]))
            .collect();
        info!("    [{}]", row.join(" "));
    }

    // Loop (favor staying in same state)
    info!("\n  Loop (self-loop prob = 0.8):");
    let loop_trans = utils::transition_loop(n_states, 0.8);
    for i in 0..n_states {
        let row: Vec<String> = (0..n_states)
            .map(|j| format!("{:.3}", loop_trans[(i, j)]))
            .collect();
        info!("    [{}]", row.join(" "));
    }

    // Local (only adjacent states)
    info!("\n  Local (only adjacent, stay prob = 0.7):");
    let local = utils::transition_local(n_states, 0.7);
    for i in 0..n_states {
        let row: Vec<String> = (0..n_states)
            .map(|j| format!("{:.3}", local[(i, j)]))
            .collect();
        info!("    [{}]", row.join(" "));
    }

    // Cycle (forward progression)
    info!("\n  Cycle (forward prob = 0.8):");
    let cycle = utils::transition_cycle(n_states, 0.8);
    for i in 0..n_states {
        let row: Vec<String> = (0..n_states)
            .map(|j| format!("{:.3}", cycle[(i, j)]))
            .collect();
        info!("    [{}]", row.join(" "));
    }
}
