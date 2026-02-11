use giggle::cqt;

#[test]
fn cqt_basic() {
    let sr: u32 = 22050;
    let signal = vec![0.1f32; sr as usize]; // 1 second

    let result = cqt::cqt(&signal, sr, 512, 110.0, 12, 12, 0.0, 1.0);
    assert!(result.is_ok());

    let cqt_matrix = result.unwrap();
    // CQT should have shape (n_bins, n_frames)
    assert!(cqt_matrix.shape()[0] > 0); // n_bins
    assert!(cqt_matrix.shape()[1] > 0); // n_frames
}

#[test]
fn cqt_empty() {
    let signal = vec![];
    let result = cqt::cqt(&signal, 22050, 512, 110.0, 12, 12, 0.0, 1.0);
    assert!(result.is_err());
}

#[test]
fn cqt_harmonic_content() {
    let sr: u32 = 22050;
    let signal = giggle::io::tone(440.0, sr, 0.5);

    let result = cqt::cqt(&signal, sr, 512, 110.0, 24, 12, 0.0, 1.0);
    assert!(result.is_ok());

    let cqt_matrix = result.unwrap();
    // Should capture energy in frequency bins
    assert!(cqt_matrix.iter().any(|&x| x.norm() > 0.0));
}

#[test]
fn icqt_reconstruction() {
    let sr: u32 = 22050;
    let signal = giggle::io::tone(440.0, sr, 0.5);
    let original_len = signal.len();

    // First compute CQT
    let cqt_matrix = cqt::cqt(&signal, sr, 512, 110.0, 12, 12, 0.0, 1.0).unwrap();

    // Reconstruct using ICQT
    let reconstructed = cqt::icqt(&cqt_matrix, sr, 512, 110.0, 12, 16).unwrap();

    // Should reconstruct some signal
    assert!(!reconstructed.is_empty());
    assert!(reconstructed.len() <= original_len * 2); // Reasonable length
}

#[test]
fn vqt_basic() {
    let sr: u32 = 22050;
    let signal = vec![0.1f32; sr as usize];

    let result = cqt::vqt(&signal, sr, 512, 110.0, 12, 12, 0.0, 1.0, 0.5);
    assert!(result.is_ok());

    let vqt_matrix = result.unwrap();
    assert!(vqt_matrix.shape()[0] > 0);
}
