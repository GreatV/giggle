use super::*;

#[test]
fn test_midi_conversions() {
    assert_eq!(midi_to_note(69), "A4");
    assert_eq!(midi_to_note(60), "C4");
    assert_eq!(note_to_midi("A4"), Some(69));
    assert_eq!(note_to_midi("C4"), Some(60));
}

#[test]
fn test_hz_conversions() {
    let hz = vec![440.0];
    let midi = hz_to_midi(&hz);
    assert!((midi[0] - 69.0).abs() < 0.01);

    let back = midi_to_hz(&midi);
    assert!((back[0] - 440.0).abs() < 0.1);
}

#[test]
fn test_time_conversions() {
    let frames = vec![0, 10, 20];
    let times = frames_to_time(&frames, 22050, 512);
    assert!((times[1] - 0.232).abs() < 0.01);

    let back = time_to_frames(&times, 22050, 512);
    assert_eq!(back, frames);
}

#[test]
fn test_mel_conversions_slaney() {
    let freqs = vec![0.0, 1000.0, 8000.0];
    let mels = hz_to_mel(&freqs, false);

    // Check monotonicity
    assert!(mels[1] > mels[0]);
    assert!(mels[2] > mels[1]);

    // Round-trip conversion
    let back = mel_to_hz(&mels, false);
    for i in 0..freqs.len() {
        assert!(
            (back[i] - freqs[i]).abs() < 1.0,
            "Expected {}, got {}",
            freqs[i],
            back[i]
        );
    }
}

#[test]
fn test_mel_conversions_htk() {
    let freqs = vec![0.0, 1000.0, 8000.0];
    let mels = hz_to_mel(&freqs, true);

    // Check monotonicity
    assert!(mels[1] > mels[0]);
    assert!(mels[2] > mels[1]);

    // Round-trip conversion
    let back = mel_to_hz(&mels, true);
    for i in 0..freqs.len() {
        assert!((back[i] - freqs[i]).abs() < 1.0);
    }
}

#[test]
fn test_octs_conversions() {
    let freqs = vec![220.0, 440.0, 880.0, 1760.0];
    let octs = hz_to_octs(&freqs, 0.0, 12);

    // A440 should be at octave 4
    assert!((octs[1] - 4.0).abs() < 0.01);

    // Each doubling should increase octave by 1
    for i in 1..octs.len() {
        assert!((octs[i] - octs[i - 1] - 1.0).abs() < 0.01);
    }

    // Round-trip
    let back = octs_to_hz(&octs, 0.0, 12);
    for i in 0..freqs.len() {
        assert!((back[i] - freqs[i]).abs() < 0.1);
    }
}

#[test]
fn test_fft_frequencies() {
    let freqs = fft_frequencies(22050, 2048);

    assert_eq!(freqs.len(), 1025); // n_fft/2 + 1
    assert_eq!(freqs[0], 0.0);

    // Check spacing
    let spacing = 22050.0 / 2048.0;
    for i in 1..freqs.len() {
        assert!((freqs[i] - freqs[i - 1] - spacing).abs() < 0.01);
    }
}

#[test]
fn test_mel_frequencies() {
    let freqs = mel_frequencies(128, 0.0, 8000.0, false);

    assert_eq!(freqs.len(), 128);
    assert!(freqs[0] >= 0.0);
    assert!(
        (freqs[127] - 8000.0).abs() < 1.0,
        "Last frequency should be close to fmax"
    );

    // Check monotonicity
    for i in 1..freqs.len() {
        assert!(freqs[i] >= freqs[i - 1], "Frequencies should be monotonic");
    }
}

#[test]
fn test_tempo_frequencies() {
    let tempos = tempo_frequencies(22050, 512, 384);

    assert!(!tempos.is_empty());
    assert_eq!(tempos[0], 0.0); // DC component

    // Should be positive and monotonic
    for i in 1..tempos.len() {
        assert!(tempos[i] > 0.0);
        assert!(tempos[i] > tempos[i - 1]);
    }
}

#[test]
fn test_mel_empty() {
    let empty: Vec<f32> = vec![];
    let mels = hz_to_mel(&empty, false);
    assert_eq!(mels.len(), 0);

    let freqs = mel_to_hz(&empty, false);
    assert_eq!(freqs.len(), 0);
}

#[test]
fn test_octs_negative_freq() {
    let freqs = vec![-1.0, 0.0];
    let octs = hz_to_octs(&freqs, 0.0, 12);

    assert!(octs[0].is_infinite() && octs[0].is_sign_negative());
    assert!(octs[1].is_infinite() && octs[1].is_sign_negative());
}

#[test]
fn test_a_weighting() {
    let freqs = vec![20.0, 100.0, 1000.0, 10000.0];
    let weights = a_weighting(&freqs, -80.0);

    assert_eq!(weights.len(), 4);

    // A-weighting should be close to 0 dB around 1 kHz
    assert!(
        weights[2].abs() < 5.0,
        "A-weighting at 1kHz should be near 0 dB, got {}",
        weights[2]
    );

    // Should be monotonic in certain ranges
    assert!(
        weights[1] > weights[0],
        "100 Hz should be weighted more than 20 Hz"
    );
}

#[test]
fn test_b_weighting() {
    let freqs = vec![100.0, 1000.0, 10000.0];
    let weights = b_weighting(&freqs, -80.0);

    assert_eq!(weights.len(), 3);

    // All weights should be finite
    for &w in &weights {
        assert!(w.is_finite());
    }
}

#[test]
fn test_c_weighting() {
    let freqs = vec![100.0, 1000.0, 10000.0];
    let weights = c_weighting(&freqs, -80.0);

    assert_eq!(weights.len(), 3);

    // C-weighting is relatively flat, so mid frequencies should have small attenuation
    assert!(weights[1] > -5.0, "C-weighting at 1kHz should be near 0 dB");
}

#[test]
fn test_d_weighting() {
    let freqs = vec![100.0, 1000.0, 10000.0];
    let weights = d_weighting(&freqs, -80.0);

    assert_eq!(weights.len(), 3);

    // All weights should be finite
    for &w in &weights {
        assert!(w.is_finite());
    }
}

#[test]
fn test_weighting_zero_freq() {
    let freqs = vec![0.0, -10.0];
    let min_db = -80.0;

    let a_w = a_weighting(&freqs, min_db);
    let b_w = b_weighting(&freqs, min_db);
    let c_w = c_weighting(&freqs, min_db);
    let d_w = d_weighting(&freqs, min_db);

    // All should return min_db for zero/negative frequencies
    assert_eq!(a_w[0], min_db);
    assert_eq!(a_w[1], min_db);
    assert_eq!(b_w[0], min_db);
    assert_eq!(c_w[0], min_db);
    assert_eq!(d_w[0], min_db);
}

#[test]
fn test_weighting_min_db_clamp() {
    let freqs = vec![10.0]; // Very low frequency with high attenuation
    let min_db = -40.0;

    let weights = a_weighting(&freqs, min_db);

    // Weight should not go below min_db
    assert!(weights[0] >= min_db);
}

#[test]
fn test_z_weighting() {
    let freqs = vec![100.0, 1000.0, 10000.0];
    let weights = z_weighting(&freqs);

    // Z-weighting should be flat (all zeros)
    for w in weights {
        assert_eq!(w, 0.0);
    }
}

#[test]
fn test_blocks_to_frames() {
    let blocks = vec![0, 1, 2, 3];
    let frames = blocks_to_frames(&blocks, 16);

    assert_eq!(frames, vec![0, 16, 32, 48]);
}

#[test]
fn test_blocks_to_samples() {
    let blocks = vec![0, 1, 2];
    let samples = blocks_to_samples(&blocks, 16, 512);

    assert_eq!(samples, vec![0, 8192, 16384]);
}

#[test]
fn test_blocks_to_time() {
    let blocks = vec![0, 1, 2];
    let times = blocks_to_time(&blocks, 16, 512, 22050);

    assert!((times[0] - 0.0).abs() < 0.001);
    assert!((times[1] - 0.372).abs() < 0.01);
}

#[test]
fn test_cqt_frequencies() {
    // 24 bins starting at C2 (~65.406 Hz)
    let freqs = cqt_frequencies(24, 65.406, 12, 0.0);

    assert_eq!(freqs.len(), 24);
    assert!((freqs[0] - 65.406).abs() < 0.01);
    // One octave up should double the frequency
    assert!((freqs[12] / freqs[0] - 2.0).abs() < 0.01);
}

#[test]
fn test_cqt_frequencies_with_tuning() {
    let freqs_no_tuning = cqt_frequencies(12, 440.0, 12, 0.0);
    let freqs_tuned = cqt_frequencies(12, 440.0, 12, 1.0);

    // Tuning should shift all frequencies up slightly
    for i in 0..12 {
        assert!(freqs_tuned[i] > freqs_no_tuning[i]);
    }
}

#[test]
fn test_fourier_tempo_frequencies() {
    let tempos = fourier_tempo_frequencies(22050, 512, 384);

    assert!(!tempos.is_empty());
    assert_eq!(tempos[0], 0.0); // DC component

    // Should be monotonic
    for i in 1..tempos.len() {
        assert!(tempos[i] > tempos[i - 1]);
    }
}

#[test]
fn test_a4_tuning_conversions() {
    // A440 should give 0 tuning
    assert!((a4_to_tuning(440.0)).abs() < 0.001);

    // Round-trip conversion
    let tuning = a4_to_tuning(442.0);
    let a4_back = tuning_to_a4(tuning);
    assert!((a4_back - 442.0).abs() < 0.01);
}

#[test]
fn test_frequency_weighting() {
    let freqs = vec![1000.0];

    let a_w = frequency_weighting(&freqs, "A", -80.0);
    let z_w = frequency_weighting(&freqs, "Z", -80.0);

    // A-weighting at 1kHz should be near 0
    assert!(a_w[0].abs() < 5.0);
    // Z-weighting should be exactly 0
    assert_eq!(z_w[0], 0.0);
}

#[test]
fn test_times_like() {
    let times = times_like(100, 22050, 512);

    assert_eq!(times.len(), 100);
    assert_eq!(times[0], 0.0);
    assert!((times[1] - 512.0 / 22050.0).abs() < 0.0001);
}

#[test]
fn test_samples_like() {
    let samples = samples_like(100, 512);

    assert_eq!(samples.len(), 100);
    assert_eq!(samples[0], 0);
    assert_eq!(samples[1], 512);
    assert_eq!(samples[99], 99 * 512);
}

// Music Theory Tests

#[test]
fn test_list_thaat() {
    let thaats = list_thaat();
    assert_eq!(thaats.len(), 10);
    assert!(thaats.contains(&"bilaval".to_string()));
    assert!(thaats.contains(&"todi".to_string()));
}

#[test]
fn test_list_mela() {
    let melas = list_mela();
    assert_eq!(melas.len(), 72);
    assert_eq!(melas[0].0, "kanakangi");
    assert_eq!(melas[0].1, 1);
    assert_eq!(melas[71].0, "rasikapriya");
    assert_eq!(melas[71].1, 72);
}

#[test]
fn test_thaat_to_degrees() {
    let bilaval = thaat_to_degrees("bilaval").unwrap();
    assert_eq!(bilaval, vec![0, 2, 4, 5, 7, 9, 11]); // Ionian/Major

    let todi = thaat_to_degrees("todi").unwrap();
    assert_eq!(todi, vec![0, 1, 3, 6, 7, 8, 11]);

    assert!(thaat_to_degrees("invalid").is_none());
}

#[test]
fn test_mela_to_degrees() {
    // Mela #1: Kanakangi
    let m1 = mela_to_degrees_by_index(1).unwrap();
    assert_eq!(m1, vec![0, 1, 2, 5, 7, 8, 9]);

    // Mela by name
    let kanakangi = mela_to_degrees("kanakangi").unwrap();
    assert_eq!(kanakangi, vec![0, 1, 2, 5, 7, 8, 9]);

    // Invalid
    assert!(mela_to_degrees_by_index(0).is_none());
    assert!(mela_to_degrees_by_index(73).is_none());
}

#[test]
fn test_mela_to_svara() {
    let svara = mela_to_svara(1, true, true);
    assert_eq!(svara.len(), 12);
    assert_eq!(svara[0], "S");
    assert_eq!(svara[7], "P");

    // Test with ASCII
    let svara_ascii = mela_to_svara(1, true, false);
    assert!(svara_ascii[1].contains("1") || svara_ascii[1].contains("R"));
}

#[test]
fn test_key_to_degrees() {
    let c_major = key_to_degrees("C:maj").unwrap();
    assert_eq!(c_major, vec![0, 2, 4, 5, 7, 9, 11]);

    let a_minor = key_to_degrees("A:min").unwrap();
    assert_eq!(a_minor, vec![9, 11, 0, 2, 4, 5, 7]);

    let d_dorian = key_to_degrees("D:dor").unwrap();
    assert_eq!(d_dorian, vec![2, 4, 5, 7, 9, 11, 0]);

    assert!(key_to_degrees("invalid").is_none());
}

#[test]
fn test_key_to_notes() {
    let c_major = key_to_notes("C:maj", true);
    assert_eq!(c_major.len(), 12);
    assert_eq!(c_major[0], "C");
    assert_eq!(c_major[1], "C♯");

    let f_major = key_to_notes("F:maj", true);
    assert_eq!(f_major[10], "B♭");

    // Test ASCII mode
    let c_major_ascii = key_to_notes("C:maj", false);
    assert_eq!(c_major_ascii[1], "C#");
}

#[test]
fn test_fifths_to_note() {
    assert_eq!(fifths_to_note("C", 0, true), "C");
    assert_eq!(fifths_to_note("C", 1, true), "G");
    assert_eq!(fifths_to_note("C", 2, true), "D");
    assert_eq!(fifths_to_note("C", 6, true), "F♯");

    // Negative fifths
    assert_eq!(fifths_to_note("C", -1, true), "F");

    // ASCII mode
    assert_eq!(fifths_to_note("C", 6, false), "F#");
}

#[test]
fn test_midi_to_svara_h() {
    // Sa = C4 (MIDI 60)
    assert_eq!(midi_to_svara_h(60, 60, true, false), "S");
    assert_eq!(midi_to_svara_h(62, 60, true, false), "R"); // Re (whole step)
    assert_eq!(midi_to_svara_h(64, 60, true, false), "G"); // Ga
    assert_eq!(midi_to_svara_h(67, 60, true, false), "P"); // Pa

    // Upper octave
    let upper = midi_to_svara_h(72, 60, true, false);
    assert!(upper.contains("S") && upper.contains("^"));
}

#[test]
fn test_hz_to_svara_h() {
    let sa = hz_to_svara_h(261.63, 261.63, true, false);
    assert_eq!(sa, "S");
}

#[test]
fn test_parse_note_to_pitch_class() {
    assert_eq!(parse_note_to_pitch_class("C"), Some(0));
    assert_eq!(parse_note_to_pitch_class("C#"), Some(1));
    assert_eq!(parse_note_to_pitch_class("Db"), Some(1));
    assert_eq!(parse_note_to_pitch_class("D"), Some(2));
    assert_eq!(parse_note_to_pitch_class("A"), Some(9));
    assert_eq!(parse_note_to_pitch_class("B"), Some(11));
}

// Interval Frequencies Tests

#[test]
fn test_interval_frequencies_equal() {
    // Equal temperament
    let freqs = interval_frequencies(12, 55.0, "equal", 12, 0.0, true);
    assert_eq!(freqs.len(), 12);
    assert!((freqs[0] - 55.0).abs() < 0.01);
    // One semitone up should be 2^(1/12) times the base
    assert!((freqs[1] / freqs[0] - 2.0f32.powf(1.0 / 12.0)).abs() < 0.001);
}

#[test]
fn test_interval_frequencies_pythagorean() {
    // Pythagorean intervals
    let freqs = interval_frequencies(24, 55.0, "pythagorean", 12, 0.0, true);
    assert_eq!(freqs.len(), 24);
    assert!((freqs[0] - 55.0).abs() < 0.01);

    // Check that frequencies are sorted
    for i in 1..freqs.len() {
        assert!(freqs[i] > freqs[i - 1]);
    }

    // One octave up should double the frequency
    assert!((freqs[12] / freqs[0] - 2.0).abs() < 0.01);
}

#[test]
fn test_interval_frequencies_ji() {
    // Just intonation intervals
    let freqs_3limit = interval_frequencies(12, 55.0, "ji3", 12, 0.0, true);
    assert_eq!(freqs_3limit.len(), 12);

    let freqs_5limit = interval_frequencies(12, 55.0, "ji5", 12, 0.0, true);
    assert_eq!(freqs_5limit.len(), 12);

    let freqs_7limit = interval_frequencies(12, 55.0, "ji7", 12, 0.0, true);
    assert_eq!(freqs_7limit.len(), 12);
}

#[test]
fn test_interval_frequencies_custom() {
    // Custom intervals
    let intervals = vec![1.0, 4.0 / 3.0, 3.0 / 2.0];
    let freqs = interval_frequencies_custom(9, 55.0, &intervals, true);
    assert_eq!(freqs.len(), 9);

    // First frequency should be fmin
    assert!((freqs[0] - 55.0).abs() < 0.01);

    // Check that frequencies are sorted
    for i in 1..freqs.len() {
        assert!(freqs[i] > freqs[i - 1]);
    }
}

#[test]
fn test_interval_frequencies_edge_cases() {
    // Invalid fmin
    let freqs = interval_frequencies(12, 0.0, "equal", 12, 0.0, true);
    assert!(freqs.is_empty());

    let freqs = interval_frequencies(12, -10.0, "equal", 12, 0.0, true);
    assert!(freqs.is_empty());

    // Zero n_bins
    let freqs = interval_frequencies(0, 55.0, "equal", 12, 0.0, true);
    assert!(freqs.is_empty());

    // Unknown interval type
    let freqs = interval_frequencies(12, 55.0, "unknown", 12, 0.0, true);
    assert!(freqs.is_empty());
}

#[test]
fn test_interval_frequencies_with_tuning() {
    let freqs_no_tuning = interval_frequencies(12, 440.0, "equal", 12, 0.0, true);
    let freqs_tuned = interval_frequencies(12, 440.0, "equal", 12, 1.0, true);

    // Tuning should shift all frequencies up
    for i in 0..12 {
        assert!(freqs_tuned[i] > freqs_no_tuning[i]);
    }
}

#[test]
fn test_interval_frequencies_no_sort() {
    let freqs_sorted = interval_frequencies(12, 55.0, "pythagorean", 12, 0.0, true);
    let freqs_unsorted = interval_frequencies(12, 55.0, "pythagorean", 12, 0.0, false);

    // Both should have same length
    assert_eq!(freqs_sorted.len(), freqs_unsorted.len());

    // Unsorted should be in circle-of-fifths order (different from sorted)
    // First element should still be 1.0 (unison)
    assert!((freqs_unsorted[0] - 55.0).abs() < 0.01);
}

#[test]
fn test_pythagorean_intervals() {
    let intervals = pythagorean_intervals(12, true, false);
    assert_eq!(intervals.len(), 12);

    // First interval should be unison (1.0)
    assert!((intervals[0] - 1.0).abs() < 1e-6);

    // Last interval should be less than 2.0 (one octave)
    assert!(intervals[11] < 2.0);

    // Check specific Pythagorean ratios
    // Perfect fifth: 3/2 = 1.5 (ascending fifths only in Pythagorean)
    let fifth = intervals.iter().find(|&&x| (x - 1.5).abs() < 0.01);
    assert!(fifth.is_some(), "Should contain perfect fifth (3/2)");

    // Major second: 9/8 = 1.125
    let major_second = intervals.iter().find(|&&x| (x - 1.125).abs() < 0.01);
    assert!(major_second.is_some(), "Should contain major second (9/8)");
}

#[test]
fn test_pythagorean_intervals_unsorted() {
    let intervals = pythagorean_intervals(7, false, false);
    assert_eq!(intervals.len(), 7);

    // First should be unison
    assert!((intervals[0] - 1.0).abs() < 1e-6);

    // Second should be perfect fifth (3/2) in circle-of-fifths order
    assert!((intervals[1] - 1.5).abs() < 0.01);
}

#[test]
fn test_pythagorean_intervals_edge_cases() {
    // Zero bins
    let intervals = pythagorean_intervals(0, true, false);
    assert!(intervals.is_empty());
}

#[test]
fn test_plimit_intervals_3limit() {
    // 3-limit (Pythagorean-like but includes negative powers)
    let intervals = plimit_intervals(&[3], 12, true, false);
    assert_eq!(intervals.len(), 12);

    // First should be unison
    assert!((intervals[0] - 1.0).abs() < 1e-6);

    // All intervals should be in [1, 2)
    for &interval in &intervals {
        assert!((1.0..2.0).contains(&interval));
    }
}

#[test]
fn test_plimit_intervals_5limit() {
    // 5-limit
    let intervals = plimit_intervals(&[3, 5], 7, true, false);
    assert_eq!(intervals.len(), 7);

    // Should contain 5/4 major third
    let major_third = intervals.iter().find(|&&x| (x - 1.25).abs() < 0.01);
    assert!(
        major_third.is_some(),
        "5-limit should contain major third (5/4)"
    );

    // Should contain 3/2 perfect fifth
    let fifth = intervals.iter().find(|&&x| (x - 1.5).abs() < 0.01);
    assert!(
        fifth.is_some(),
        "5-limit should contain perfect fifth (3/2)"
    );
}

#[test]
fn test_plimit_intervals_7limit() {
    // 7-limit
    let intervals = plimit_intervals(&[3, 5, 7], 12, true, false);
    assert_eq!(intervals.len(), 12);

    // Should contain 7/4 harmonic seventh
    let harmonic_seventh = intervals.iter().find(|&&x| (x - 1.75).abs() < 0.01);
    assert!(
        harmonic_seventh.is_some(),
        "7-limit should contain harmonic seventh (7/4)"
    );
}

#[test]
fn test_plimit_intervals_unsorted() {
    let intervals = plimit_intervals(&[3, 5], 7, false, false);
    assert_eq!(intervals.len(), 7);

    // First should be unison
    assert!((intervals[0] - 1.0).abs() < 1e-6);
}

#[test]
fn test_plimit_intervals_edge_cases() {
    // Empty primes
    let intervals = plimit_intervals(&[], 12, true, false);
    assert!(intervals.is_empty());

    // Zero bins
    let intervals = plimit_intervals(&[3], 0, true, false);
    assert!(intervals.is_empty());

    // Even primes should be filtered out (only odd primes allowed)
    let intervals = plimit_intervals(&[2], 12, true, false);
    assert!(intervals.is_empty());
}

#[test]
fn test_plimit_vs_pythagorean() {
    // 3-limit should be different from Pythagorean
    // Pythagorean only uses positive powers of 3
    // 3-limit uses both positive and negative powers
    let plimit_3 = plimit_intervals(&[3], 12, true, false);
    let pythagorean = pythagorean_intervals(12, true, false);

    // Both should have 12 intervals
    assert_eq!(plimit_3.len(), 12);
    assert_eq!(pythagorean.len(), 12);

    // They should be different (at least some intervals differ)
    let all_same = plimit_3
        .iter()
        .zip(pythagorean.iter())
        .all(|(a, b)| (a - b).abs() < 0.001);
    assert!(!all_same, "3-limit and Pythagorean intervals should differ");
}
