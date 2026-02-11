use num_complex::Complex32;
use realfft::RealFftPlanner;
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;

/// FFT plan for forward and inverse complex FFT operations.
///
/// This struct caches FFT plans for efficient reuse. The underlying planner
/// uses the Cooley-Tukey algorithm and is optimized via the rustfft library.
///
/// # Example
/// ```
/// use giggle::fft::FftPlan;
/// use num_complex::Complex32;
///
/// let plan = FftPlan::new(512);
/// let mut buffer = vec![Complex32::new(1.0, 0.0); 512];
/// plan.forward(&mut buffer);
/// plan.inverse(&mut buffer);
/// ```
pub struct FftPlan {
    forward: Arc<dyn Fft<f32>>,
    inverse: Arc<dyn Fft<f32>>,
    len: usize,
}

impl FftPlan {
    /// Create a new FFT plan for a given size.
    ///
    /// # Arguments
    /// * `len` - Size of the FFT (must be a power of 2 for best performance)
    ///
    /// # Returns
    /// A new FftPlan instance
    pub fn new(len: usize) -> Self {
        let mut planner = FftPlanner::new();
        let forward = planner.plan_fft_forward(len);
        let inverse = planner.plan_fft_inverse(len);
        Self {
            forward,
            inverse,
            len,
        }
    }

    /// Perform forward FFT in-place.
    ///
    /// # Arguments
    /// * `buffer` - Complex input buffer, will be overwritten with FFT output
    pub fn forward(&self, buffer: &mut [Complex32]) {
        self.forward.process(buffer);
    }

    /// Perform inverse FFT in-place.
    ///
    /// The output is scaled by 1/len to make the transform orthogonal.
    ///
    /// # Arguments
    /// * `buffer` - Complex input buffer, will be overwritten with IFFT output
    pub fn inverse(&self, buffer: &mut [Complex32]) {
        self.inverse.process(buffer);
        let scale = 1.0 / self.len as f32;
        for v in buffer.iter_mut() {
            *v *= scale;
        }
    }
}

#[cfg(feature = "parallel")]
const _: () = {
    fn _assert_send_sync<T: Send + Sync>() {}
    fn _check() {
        _assert_send_sync::<FftPlan>();
    }
};

/// Compute the real-to-complex FFT (rfft) of a real-valued input.
///
/// This function computes the FFT of real input data, returning only the
/// non-redundant half of the spectrum (due to symmetry for real inputs).
///
/// # Arguments
/// * `input` - Real-valued input signal
///
/// # Returns
/// Complex FFT output of length input.len() / 2 + 1
///
/// # Example
/// ```
/// use giggle::fft::rfft;
///
/// let signal = vec![1.0f32; 1024];
/// let spectrum = rfft(&signal);
/// assert_eq!(spectrum.len(), 513); // 1024/2 + 1
/// ```
pub fn rfft(input: &[f32]) -> Vec<Complex32> {
    if input.is_empty() {
        return Vec::new();
    }
    let len = input.len();
    let mut planner = RealFftPlanner::<f32>::new();
    let r2c = planner.plan_fft_forward(len);
    let mut in_buf = input.to_vec();
    let mut out_buf = r2c.make_output_vec();
    let _ = r2c.process(&mut in_buf, &mut out_buf);
    out_buf
}
