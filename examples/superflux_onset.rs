//! Superflux Onset Detection Example
//!
//! This example demonstrates the Superflux onset detection algorithm of
//! Boeck and Widmer, 2013. This algorithm improves onset detection accuracy
//! in the presence of vibrato.
//!
//! Based on librosa's plot_superflux.py example.

use giggle::feature::mel::melspectrogram;
use giggle::io;
use giggle::onset::detect::onset_detect;
use giggle::onset::strength::onset_strength;
use log::info;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    info!("Superflux Onset Detection Example");

    let sr = 22050;

    // Generate test signal with vibrato
    info!("Generating Test Signal");

    // Create a signal with vibrato (frequency modulation)
    let duration = 3.0;
    let base_freq = 440.0;
    let vibrato_rate = 5.0; // 5 Hz vibrato
    let vibrato_depth = 10.0; // +/- 10 Hz

    let n_samples = (duration * sr as f32) as usize;
    let mut signal = vec![0.0f32; n_samples];

    for (i, s) in signal.iter_mut().enumerate().take(n_samples) {
        let t = i as f32 / sr as f32;
        // Frequency modulation for vibrato
        let freq =
            base_freq + vibrato_depth * (2.0 * std::f32::consts::PI * vibrato_rate * t).sin();
        let phase = 2.0 * std::f32::consts::PI * freq * t;
        *s = phase.sin() * 0.5;
    }

    // Add some clear onsets
    let onset_times = [0.5, 1.0, 1.5, 2.0, 2.5];
    for onset_time in &onset_times {
        let start = (*onset_time * sr as f32) as usize;
        let click = io::tone(880.0, sr, 0.05);
        for (i, sample) in click.iter().enumerate() {
            if start + i < signal.len() {
                signal[start + i] += sample * 0.3;
            }
        }
    }

    info!("Generated {} samples ({:.1}s)", signal.len(), duration);
    info!("  - Base frequency: {} Hz with vibrato", base_freq);
    info!(
        "  - Vibrato: {} Hz rate, +/- {} Hz depth",
        vibrato_rate, vibrato_depth
    );
    info!(
        "  - Added {} clear onsets at: {:?}",
        onset_times.len(),
        onset_times
    );

    // Standard Onset Detection
    info!("Standard Onset Detection");

    let hop_length = 512;
    let n_fft = 2048;

    let onset_env_default = onset_strength(&signal, n_fft, hop_length)?;
    let onsets_default = onset_detect(&signal, n_fft, hop_length, 0.1)?;

    info!(
        "Onset strength envelope: {} frames",
        onset_env_default.len()
    );
    info!("Detected {} onsets", onsets_default.len());

    let onset_times_default: Vec<f32> = onsets_default
        .iter()
        .map(|&frame| frame as f32 * hop_length as f32 / sr as f32)
        .collect();

    info!("Onset times: {:?}", onset_times_default);

    // Superflux Onset Detection
    info!("Superflux Onset Detection");

    // Superflux parameters from the paper
    let n_fft = 1024;
    let n_mels = 138;
    let _fmin = 27.5;
    let _fmax = 8000.0f32.min(sr as f32 / 2.0);
    let lag = 2;
    let max_size = 3;

    // Compute mel spectrogram
    let mel_spec = melspectrogram(&signal, sr, n_fft, hop_length, n_mels)?;
    info!("Mel spectrogram shape: {:?}", mel_spec.shape());

    // Convert to dB
    let mut mel_db = mel_spec.clone();
    for val in mel_db.iter_mut() {
        *val = 10.0 * ((*val + 1e-10).log10());
    }

    // Compute Superflux onset strength
    let onset_env_superflux = compute_superflux(&mel_db, lag, max_size);

    info!(
        "Superflux onset envelope: {} frames",
        onset_env_superflux.len()
    );

    // Detect onsets from Superflux envelope
    // Detect onsets from Superflux envelope manually
    let threshold =
        onset_env_superflux.iter().sum::<f32>() / onset_env_superflux.len() as f32 * 1.5;
    let mut onsets_superflux = Vec::new();
    for i in 1..onset_env_superflux.len().saturating_sub(1) {
        if onset_env_superflux[i] > threshold
            && onset_env_superflux[i] >= onset_env_superflux[i - 1]
            && onset_env_superflux[i] > onset_env_superflux[i + 1]
        {
            onsets_superflux.push(i);
        }
    }
    info!("Detected {} onsets (Superflux)", onsets_superflux.len());

    let onset_times_superflux: Vec<f32> = onsets_superflux
        .iter()
        .map(|&frame| frame as f32 * hop_length as f32 / sr as f32)
        .collect();

    info!("Onset times (Superflux): {:?}", onset_times_superflux);

    // Comparison
    info!("Comparison");

    info!("Method          | Onsets | Times (s)");
    info!(
        "Standard        | {:6} | {:?}",
        onsets_default.len(),
        onset_times_default
    );
    info!(
        "Superflux       | {:6} | {:?}",
        onsets_superflux.len(),
        onset_times_superflux
    );
    info!(
        "Expected        | {:6} | {:?}",
        onset_times.len(),
        onset_times
    );

    // Compute envelope statistics
    let mean_default: f32 = onset_env_default.iter().sum::<f32>() / onset_env_default.len() as f32;
    let max_default = onset_env_default.iter().fold(0.0f32, |a, &b| a.max(b));

    let mean_superflux: f32 =
        onset_env_superflux.iter().sum::<f32>() / onset_env_superflux.len() as f32;
    let max_superflux = onset_env_superflux.iter().fold(0.0f32, |a, &b| a.max(b));

    info!("\nEnvelope statistics:");
    info!(
        "  Standard:  mean={:.2}, max={:.2}",
        mean_default, max_default
    );
    info!(
        "  Superflux: mean={:.2}, max={:.2}",
        mean_superflux, max_superflux
    );

    // Vibrato Robustness Test
    info!("Vibrato Robustness Test");

    // Create a signal with strong vibrato and fewer clear onsets
    let mut vibrato_signal = vec![0.0f32; sr as usize * 2]; // 2 seconds
    for (i, s) in vibrato_signal.iter_mut().enumerate() {
        let t = i as f32 / sr as f32;
        // Strong vibrato
        let freq = 440.0 + 30.0 * (2.0 * std::f32::consts::PI * 6.0 * t).sin();
        *s = (2.0 * std::f32::consts::PI * freq * t).sin() * 0.5;
    }

    // Add one clear onset
    let click = io::tone(1000.0, sr, 0.1);
    for (i, sample) in click.iter().enumerate() {
        if i < vibrato_signal.len() {
            vibrato_signal[i] += sample * 0.5;
        }
    }

    let env_default = onset_strength(&vibrato_signal, n_fft, hop_length)?;
    let env_superflux = compute_superflux_from_signal(
        &vibrato_signal,
        sr,
        n_fft,
        hop_length,
        n_mels,
        lag,
        max_size,
    )?;

    // Count peaks above threshold
    let threshold_default = env_default.iter().sum::<f32>() / env_default.len() as f32 * 2.0;
    let threshold_superflux = env_superflux.iter().sum::<f32>() / env_superflux.len() as f32 * 2.0;

    let peaks_default = count_peaks(&env_default, threshold_default);
    let peaks_superflux = count_peaks(&env_superflux, threshold_superflux);

    info!("Strong vibrato signal (1 clear onset):");
    info!(
        "  Standard method peaks:  {} (threshold: {:.2})",
        peaks_default, threshold_default
    );
    info!(
        "  Superflux method peaks: {} (threshold: {:.2})",
        peaks_superflux, threshold_superflux
    );
    info!(
        "  Superflux is {} robust to vibrato",
        if peaks_superflux < peaks_default {
            "more"
        } else {
            "less"
        }
    );

    Ok(())
}

/// Compute Superflux onset strength envelope
fn compute_superflux(mel_spec: &ndarray::Array2<f32>, lag: usize, max_size: usize) -> Vec<f32> {
    let n_mels = mel_spec.shape()[0];
    let n_frames = mel_spec.shape()[1];

    let mut onset_env = Vec::with_capacity(n_frames);

    for t in 0..n_frames {
        if t < lag {
            onset_env.push(0.0);
            continue;
        }

        let mut flux = 0.0f32;

        for m in 0..n_mels {
            // Apply maximum filter (suppress vibrato)
            let max_val = ((m.saturating_sub(max_size))..=(m + max_size).min(n_mels - 1))
                .map(|mm| mel_spec[(mm, t.saturating_sub(lag))])
                .fold(f32::NEG_INFINITY, f32::max);

            let diff = mel_spec[(m, t)] - max_val;
            if diff > 0.0 {
                flux += diff;
            }
        }

        onset_env.push(flux);
    }

    onset_env
}

/// Compute Superflux from raw signal
fn compute_superflux_from_signal(
    signal: &[f32],
    sr: u32,
    n_fft: usize,
    hop_length: usize,
    n_mels: usize,
    lag: usize,
    max_size: usize,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let mel_spec = melspectrogram(signal, sr, n_fft, hop_length, n_mels)?;

    // Convert to dB
    let mut mel_db = mel_spec;
    for val in mel_db.iter_mut() {
        *val = 10.0 * ((*val + 1e-10).log10());
    }

    Ok(compute_superflux(&mel_db, lag, max_size))
}

/// Count peaks above threshold
fn count_peaks(envelope: &[f32], threshold: f32) -> usize {
    let mut count = 0;
    for i in 1..envelope.len().saturating_sub(1) {
        if envelope[i] > threshold && envelope[i] > envelope[i - 1] && envelope[i] > envelope[i + 1]
        {
            count += 1;
        }
    }
    count
}
