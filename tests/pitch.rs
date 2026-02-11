use giggle::pitch;

#[test]
fn pitch_yin_pure_tone() {
    let sr = 22050;
    let freq = 440.0;
    let signal = giggle::io::tone(freq, sr, 0.5);

    let f0 = pitch::yin(&signal, sr, 2048, 512, 40.0, 5000.0, 0.1).unwrap();

    // Should detect pitch in some frames
    let detected: Vec<f32> = f0.iter().filter(|&&x| x > 0.0).cloned().collect();
    assert!(!detected.is_empty());

    // Average frequency should be close to 440 Hz
    if !detected.is_empty() {
        let avg_f0 = detected.iter().sum::<f32>() / detected.len() as f32;
        assert!((avg_f0 - freq).abs() < 100.0);
    }
}

#[test]
fn pitch_yin_empty() {
    let signal = vec![];
    let result = pitch::yin(&signal, 22050, 2048, 512, 40.0, 5000.0, 0.1);
    assert!(result.is_err());
    assert!(matches!(result, Err(giggle::Error::EmptyAudio)));
}

#[test]
fn pitch_yin_invalid_params() {
    let signal = vec![0.1f32; 1000];

    // Invalid frame_length
    let result = pitch::yin(&signal, 22050, 0, 512, 40.0, 5000.0, 0.1);
    assert!(result.is_err());

    // Invalid hop_length
    let result = pitch::yin(&signal, 22050, 2048, 0, 40.0, 5000.0, 0.1);
    assert!(result.is_err());
}

#[test]
fn pitch_piptrack_basic() {
    let sr = 22050;
    let freq = 440.0;
    let signal = giggle::io::tone(freq, sr, 0.5);

    let (pitches, _mags) = pitch::piptrack(&signal, sr, 2048, 512, 150.0, 4000.0, 0.01).unwrap();

    // Should have output
    assert!(pitches.shape()[0] > 0);
    assert!(pitches.shape()[1] > 0);

    // Some pitches should be detected
    let detected_count = pitches.iter().filter(|&&x| x > 0.0).count();
    assert!(detected_count > 0);
}

#[test]
fn pitch_pyin_basic() {
    let sr = 22050;
    let signal = giggle::io::tone(440.0, sr, 0.5);

    let (f0, voiced, probs) =
        pitch::pyin(&signal, sr, 65.0, 2093.0, 2048, 512, 100, (2.0, 18.0), 0.1);

    // Should have output
    assert_eq!(f0.len(), voiced.len());
    assert_eq!(f0.len(), probs.len());

    // At least some frames should be voiced
    let voiced_count = voiced.iter().filter(|&&v| v).count();
    assert!(voiced_count > 0);
}
