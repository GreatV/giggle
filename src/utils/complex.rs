/// Compute squared magnitude (|x|^2) for complex numbers.
///
/// # Arguments
/// * `x` - Complex number
///
/// # Returns
/// |x|^2 = x.re^2 + x.im^2
pub fn abs2(x: num_complex::Complex32) -> f32 {
    x.re * x.re + x.im * x.im
}

/// Convert phase angle to unit complex number (phasor).
///
/// # Arguments
/// * `angle` - Phase angle in radians
///
/// # Returns
/// Complex number e^(i*angle) = cos(angle) + i*sin(angle)
pub fn phasor(angle: f32) -> num_complex::Complex32 {
    num_complex::Complex32::new(angle.cos(), angle.sin())
}
