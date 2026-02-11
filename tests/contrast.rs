use giggle::feature::contrast;

#[test]
fn spectral_contrast_shape() {
    let sr = 22050u32;
    let y = vec![0.0f32; sr as usize / 2];
    let out = contrast::spectral_contrast(&y, sr, 512, 128, 6, 200.0).unwrap();
    assert_eq!(out.shape()[0], 7);
}

#[test]
fn spectral_contrast_empty() {
    let result = contrast::spectral_contrast(&[], 22050, 512, 128, 6, 200.0);
    assert!(result.is_err());
    assert!(matches!(result, Err(giggle::Error::EmptyAudio)));
}

#[test]
fn spectral_contrast_invalid_fmin() {
    let y = vec![0.0f32; 1000];
    let result = contrast::spectral_contrast(&y, 22050, 512, 128, 6, 0.0);
    assert!(result.is_err());
}

#[test]
fn spectral_contrast_invalid_n_fft() {
    let y = vec![0.0f32; 1000];
    let result = contrast::spectral_contrast(&y, 22050, 0, 128, 6, 200.0);
    assert!(result.is_err());
}
