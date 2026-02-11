//! Music Synthesis Example
//!
//! This example demonstrates generating various musical signals using giggle's
//! signal generators: tones, chirps, clicks, and combining them to create
//! rhythmic patterns and musical examples.

use giggle::io;
use log::info;

fn main() {
    env_logger::init();
    info!("Music Synthesis Example");

    let sr = 22050;

    // Part 1: Pure Tones
    info!("Part 1: Pure Tones");

    // Generate tones at different frequencies
    let frequencies = [220.0, 440.0, 880.0]; // A3, A4, A5
    let duration = 1.0;

    for freq in &frequencies {
        let tone = io::tone(*freq, sr, duration);
        info!(
            "Generated {:.1} Hz tone (A{}): {} samples",
            freq,
            (69.0 + 12.0 * (*freq / 440.0).log2()).round() as i32 - 69 + 4,
            tone.len()
        );

        // Calculate RMS energy
        let rms: f32 = (tone.iter().map(|s| s * s).sum::<f32>() / tone.len() as f32).sqrt();
        info!("  RMS energy: {:.4}", rms);
    }

    // Part 2: Musical Chords
    info!("Part 2: Musical Chords");

    // C Major chord: C4 (261.63), E4 (329.63), G4 (392.00)
    info!("C Major chord (C4, E4, G4):");
    let c4 = io::tone(261.63, sr, 2.0);
    let e4 = io::tone(329.63, sr, 2.0);
    let g4 = io::tone(392.00, sr, 2.0);

    let c_major: Vec<f32> = c4
        .iter()
        .zip(e4.iter())
        .zip(g4.iter())
        .map(|((c, e), g)| (c + e + g) / 3.0)
        .collect();

    info!("  Mixed {} samples", c_major.len());

    // A minor chord: A3 (220.00), C4 (261.63), E4 (329.63)
    info!("\nA minor chord (A3, C4, E4):");
    let a3 = io::tone(220.00, sr, 2.0);
    let c4 = io::tone(261.63, sr, 2.0);
    let e4 = io::tone(329.63, sr, 2.0);

    let a_minor: Vec<f32> = a3
        .iter()
        .zip(c4.iter())
        .zip(e4.iter())
        .map(|((a, c), e)| (a + c + e) / 3.0)
        .collect();

    info!("  Mixed {} samples", a_minor.len());

    // Part 3: Chirps (Frequency Sweeps)
    info!("Part 3: Chirps (Frequency Sweeps)");

    // Logarithmic chirp (default) - exponential frequency sweep
    let f0 = 100.0;
    let f1 = 1000.0;
    let chirp_duration = 2.0;

    let log_chirp = io::chirp(f0, f1, sr, chirp_duration);
    info!("Logarithmic chirp: {} Hz to {} Hz", f0, f1);
    info!(
        "  Duration: {:.1}s, Samples: {}",
        chirp_duration,
        log_chirp.len()
    );

    // Linear chirp
    let linear_chirp = io::chirp_with_mode(f0, f1, sr, chirp_duration, true);
    info!("\nLinear chirp: {} Hz to {} Hz", f0, f1);
    info!(
        "  Duration: {:.1}s, Samples: {}",
        chirp_duration,
        linear_chirp.len()
    );

    // Part 4: Rhythmic Patterns with Clicks
    info!("Part 4: Rhythmic Patterns with Clicks");

    // Create a simple drum pattern (4/4 time, 120 BPM)
    let bpm = 120.0;
    let beats_per_second = bpm / 60.0;
    let seconds_per_beat = 1.0 / beats_per_second;
    let pattern_duration = 4.0; // 4 bars

    info!("Drum pattern at {} BPM (4/4 time, 4 bars)", bpm as i32);

    // Kick drum on beats 1, 2, 3, 4
    let kick_times: Vec<f32> = (0..16).map(|i| i as f32 * seconds_per_beat).collect();

    let kick = io::clicks(
        &kick_times,
        sr,
        Some((pattern_duration * sr as f32) as usize),
        0.05, // click duration
        60.0, // low frequency for kick
    );

    info!("  Kick drum: {} hits", kick_times.len());

    // Hi-hat on off-beats (8th notes)
    let hihat_times: Vec<f32> = (0..32).map(|i| i as f32 * seconds_per_beat / 2.0).collect();

    let hihat = io::clicks(
        &hihat_times,
        sr,
        Some((pattern_duration * sr as f32) as usize),
        0.02,   // shorter duration
        8000.0, // high frequency for hi-hat
    );

    info!("  Hi-hat: {} hits", hihat_times.len());

    // Snare on beats 2 and 4
    let snare_times: Vec<f32> = (0..8)
        .map(|i| (i as f32 * 2.0 + 1.0) * seconds_per_beat)
        .collect();

    let snare = io::clicks(
        &snare_times,
        sr,
        Some((pattern_duration * sr as f32) as usize),
        0.03,
        200.0, // mid frequency for snare
    );

    info!("  Snare: {} hits", snare_times.len());

    // Mix the drum pattern
    let drum_pattern: Vec<f32> = kick
        .iter()
        .zip(hihat.iter())
        .zip(snare.iter())
        .map(|((k, h), s)| k * 0.6 + h * 0.3 + s * 0.5)
        .collect();

    info!(
        "\n  Mixed drum pattern: {} samples ({:.2}s)",
        drum_pattern.len(),
        drum_pattern.len() as f32 / sr as f32
    );

    // Calculate energy per beat
    info!("\n  Energy per beat:");
    let samples_per_beat = (seconds_per_beat * sr as f32) as usize;
    for beat in 0..4 {
        let start = beat * samples_per_beat;
        let end = ((beat + 1) * samples_per_beat).min(drum_pattern.len());
        if start < drum_pattern.len() {
            let energy: f32 = drum_pattern[start..end].iter().map(|s| s * s).sum();
            info!("    Beat {}: {:.4}", beat + 1, energy);
        }
    }

    // Part 5: Melodic Sequence
    info!("Part 5: Melodic Sequence");

    // Create a simple melody (C major scale: C-D-E-F-G-A-B-C)
    let scale_degrees = [
        261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25,
    ];
    let note_duration = 0.5;
    let _note_samples = (note_duration * sr as f32) as usize;

    info!("C major scale melody:");
    let note_names = ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5"];

    let mut melody = Vec::new();
    for (i, freq) in scale_degrees.iter().enumerate() {
        let tone = io::tone(*freq, sr, note_duration);
        melody.extend_from_slice(&tone);
        info!(
            "  {} ({:.2} Hz): {} samples",
            note_names[i],
            freq,
            tone.len()
        );
    }

    info!(
        "\n  Total melody: {} samples ({:.2}s)",
        melody.len(),
        melody.len() as f32 / sr as f32
    );

    // Part 6: Complex Sound Design
    info!("Part 6: Complex Sound Design");

    // Create a "riser" effect: rising pitch + increasing volume + filter sweep simulation
    let riser_duration = 3.0;

    // Rising pitch chirp
    let riser_pitch = io::chirp(100.0, 1000.0, sr, riser_duration);

    // Add some harmonic content by mixing octaves
    let riser_octave = io::chirp(200.0, 2000.0, sr, riser_duration);

    // Apply amplitude envelope (fade in then fade out)
    let riser: Vec<f32> = riser_pitch
        .iter()
        .zip(riser_octave.iter())
        .enumerate()
        .map(|(i, (p, o))| {
            let t = i as f32 / sr as f32;
            // Envelope: fade in for 20%, sustain 60%, fade out for 20%
            let envelope = if t < riser_duration * 0.2 {
                t / (riser_duration * 0.2)
            } else if t > riser_duration * 0.8 {
                (riser_duration - t) / (riser_duration * 0.2)
            } else {
                1.0
            };
            (p + o * 0.5) * envelope * 0.5
        })
        .collect();

    info!("  Total samples: {}", riser.len());

    // Calculate RMS energy over time (showing the envelope)
    info!("\n  RMS energy profile (100ms windows):");
    let window_size = sr as usize / 10; // 100ms windows
    for i in 0..10 {
        let start = i * window_size;
        let end = ((i + 1) * window_size).min(riser.len());
        if start < riser.len() {
            let window = &riser[start..end];
            let rms = (window.iter().map(|s| s * s).sum::<f32>() / window.len() as f32).sqrt();
            let bar = "█".repeat((rms * 20.0) as usize);
            info!("    {:.1}s: {:.3} {}", i as f32 * 0.1, rms, bar);
        }
    }

    // Part 7: Audio Processing Utilities
    info!("Part 7: Audio Processing Utilities");

    // Generate a test signal
    let test_signal = io::tone(440.0, sr, 1.0);

    // Mu-law compression (used in telephony)
    let mu = 255.0;
    let compressed = io::mu_compress(&test_signal, mu);
    let expanded = io::mu_expand(&compressed, mu);

    info!("Mu-law compression (μ = {}):", mu as i32);
    info!(
        "  Original range: [{:.3}, {:.3}]",
        test_signal.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
        test_signal.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b))
    );
    info!(
        "  Compressed range: [{:.3}, {:.3}]",
        compressed.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
        compressed.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b))
    );

    // Verify round-trip
    let max_error: f32 = test_signal
        .iter()
        .zip(expanded.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    info!("  Max reconstruction error: {:.2e}", max_error);

    // LPC analysis
    info!("\nLPC analysis of 440 Hz tone:");
    let lpc_order = 10;
    let lpc_coeffs = io::lpc(&test_signal, lpc_order);
    info!("  Order: {}", lpc_order);
    info!(
        "  First 5 coefficients: {:.4}, {:.4}, {:.4}, {:.4}, {:.4}",
        lpc_coeffs[0], lpc_coeffs[1], lpc_coeffs[2], lpc_coeffs[3], lpc_coeffs[4]
    );
}
