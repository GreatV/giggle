use giggle::beat;

#[test]
fn beat_track_basic() {
    let sr: u32 = 22050;
    // Create a signal with periodic beats (clicks every 0.5 seconds)
    let mut signal = vec![0.0f32; sr as usize]; // 1 second of audio
    let beat_interval = (sr as f32 / 2.0) as usize; // 120 BPM = 2 beats per second
    for i in (beat_interval..sr as usize).step_by(beat_interval) {
        if i < signal.len() {
            signal[i] = 1.0;
        }
    }

    let result = beat::beat_track(&signal, sr, None, 512, Some(120.0));
    assert!(result.is_ok());
    let (_tempo, beats) = result.unwrap();

    // Should detect some beats
    assert!(!beats.is_empty());
}

#[test]
fn beat_track_empty() {
    let signal = vec![];
    let result = beat::beat_track(&signal, 22050, None, 512, None);
    assert!(result.is_ok());
    let (tempo, beats) = result.unwrap();
    assert_eq!(beats.len(), 0);
    assert_eq!(tempo, 120.0); // Default tempo
}

#[test]
fn beat_track_silence() {
    let signal = vec![0.0f32; 22050];
    let result = beat::beat_track(&signal, 22050, None, 512, None);
    assert!(result.is_ok());
    let (_tempo, beats) = result.unwrap();
    // Silence should produce very few or no beats
    assert!(beats.len() < 10);
}

#[test]
fn beat_track_with_onset_envelope() {
    let sr: u32 = 22050;
    let signal = vec![0.1f32; sr as usize]; // 1 second of low-amplitude noise

    // Compute onset envelope first
    let onset_env = giggle::onset::strength::onset_strength(&signal, 2048, 512).unwrap();

    let result = beat::beat_track(&signal, sr, Some(&onset_env), 512, Some(120.0));
    assert!(result.is_ok());
}
