use giggle::feature::{mel, mfcc};

#[test]
fn mel_filterbank_shapes() {
    let fb = mel::mel_filterbank(22050, 1024, 40, 0.0, 11025.0);
    assert_eq!(fb.shape(), &[40, 513]);

    for m in 0..40 {
        let mut sum = 0.0f32;
        for f in 0..513 {
            sum += fb[(m, f)];
        }
        assert!(sum > 0.0, "mel filter sum should be positive");
    }
}

/// Validate that mel filterbank uses slaney normalization (area normalization).
///
/// In slaney normalization, each triangular filter is scaled by 2 / (f_upper - f_lower),
/// which makes the area under each filter approximately equal to 2.
#[test]
fn mel_filterbank_slaney_normalization() {
    let sr = 22050u32;
    let n_fft = 2048usize;
    let n_mels = 40usize;
    let fb = mel::mel_filterbank(sr, n_fft, n_mels, 0.0, sr as f32 / 2.0);

    let n_freq = n_fft / 2 + 1;
    let freq_resolution = sr as f32 / n_fft as f32;

    // For slaney normalization, the area under each filter should be approximately 2
    // Area ≈ sum(filter_values) * freq_resolution
    // The tolerance is generous because of discrete sampling effects
    for m in 0..n_mels {
        let mut area = 0.0f64;
        for f in 0..n_freq {
            area += fb[(m, f)] as f64 * freq_resolution as f64;
        }
        // Area should be approximately 2 for slaney normalization
        // (with some tolerance for discrete sampling)
        assert!(
            area > 0.5 && area < 5.0,
            "mel filter {m} area {area} out of expected range for slaney normalization"
        );
    }

    // Additionally verify that all filters have a peak value
    for m in 0..n_mels {
        let mut max_val = 0.0f32;
        for f in 0..n_freq {
            max_val = max_val.max(fb[(m, f)]);
        }
        assert!(max_val > 0.0, "mel filter {m} has no peak value");
    }
}

/// Test mel filterbank triangular shape property.
///
/// Each filter should have a single peak (triangular shape).
#[test]
fn mel_filterbank_triangular_shape() {
    let fb = mel::mel_filterbank(22050, 2048, 40, 0.0, 11025.0);
    let n_freq = 1025;

    for m in 0..40 {
        // Find the peak
        let mut peak_idx = 0usize;
        let mut peak_val = 0.0f32;
        for f in 0..n_freq {
            if fb[(m, f)] > peak_val {
                peak_val = fb[(m, f)];
                peak_idx = f;
            }
        }

        // Verify monotonic increase before peak and monotonic decrease after
        // (with some tolerance for numerical precision)
        let mut prev = 0.0f32;
        for f in 0..peak_idx {
            let val = fb[(m, f)];
            assert!(
                val >= prev - 1e-6,
                "mel filter {m} not monotonically increasing before peak at bin {f}"
            );
            prev = val;
        }

        prev = peak_val;
        for f in peak_idx..n_freq {
            let val = fb[(m, f)];
            assert!(
                val <= prev + 1e-6,
                "mel filter {m} not monotonically decreasing after peak at bin {f}"
            );
            prev = val;
        }
    }
}

/// Test mel frequency conversion consistency.
#[test]
fn mel_hz_conversion_roundtrip() {
    let test_freqs = [0.0, 100.0, 440.0, 1000.0, 4000.0, 8000.0, 16000.0];

    for &hz in &test_freqs {
        let mel_val = mel::hz_to_mel(hz);
        let hz_back = mel::mel_to_hz(mel_val);
        assert!(
            (hz - hz_back).abs() < 0.01,
            "Hz-Mel-Hz roundtrip failed for {hz}: got {hz_back}"
        );
    }
}

/// Test mel frequencies are monotonically increasing.
#[test]
fn mel_frequencies_monotonic() {
    let freqs = mel::mel_frequencies(128, 0.0, 22050.0);

    for i in 1..freqs.len() {
        assert!(
            freqs[i] > freqs[i - 1],
            "mel frequencies not monotonically increasing at index {i}"
        );
    }
}

#[test]
fn mfcc_shape() {
    let sr = 22050u32;
    let y = vec![0.0f32; sr as usize / 2];
    let out = mfcc::mfcc(&y, sr, 13, 512, 128, 40).unwrap();
    assert_eq!(out.shape()[0], 13);
}
