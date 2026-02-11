/// Convert frame indices to time (seconds).
pub fn frames_to_time(frames: &[usize], sr: u32, hop_length: usize) -> Vec<f32> {
    frames
        .iter()
        .map(|&f| (f * hop_length) as f32 / sr as f32)
        .collect()
}

/// Convert time (seconds) to frame indices.
pub fn time_to_frames(times: &[f32], sr: u32, hop_length: usize) -> Vec<usize> {
    times
        .iter()
        .map(|&t| ((t * sr as f32) / hop_length as f32).round() as usize)
        .collect()
}

/// Convert frame indices to sample indices.
pub fn frames_to_samples(frames: &[usize], hop_length: usize) -> Vec<usize> {
    frames.iter().map(|&f| f * hop_length).collect()
}

/// Convert sample indices to frame indices.
pub fn samples_to_frames(samples: &[usize], hop_length: usize) -> Vec<usize> {
    samples
        .iter()
        .map(|&s| (s as f32 / hop_length as f32).round() as usize)
        .collect()
}

/// Convert time (seconds) to samples.
pub fn time_to_samples(times: &[f32], sr: u32) -> Vec<usize> {
    times
        .iter()
        .map(|&t| (t * sr as f32).round() as usize)
        .collect()
}

/// Convert samples to time (seconds).
pub fn samples_to_time(samples: &[usize], sr: u32) -> Vec<f32> {
    samples.iter().map(|&s| s as f32 / sr as f32).collect()
}

/// Convert block indices to frame indices.
///
/// # Arguments
/// * `blocks` - Block index or array of block indices
/// * `block_length` - Number of frames per block
///
/// # Returns
/// Frame indices corresponding to the beginning of each block
///
/// # Example
/// ```
/// use giggle::convert::blocks_to_frames;
///
/// let frames = blocks_to_frames(&[0, 1, 2], 16);
/// assert_eq!(frames, vec![0, 16, 32]);
/// ```
pub fn blocks_to_frames(blocks: &[usize], block_length: usize) -> Vec<usize> {
    blocks.iter().map(|&b| b * block_length).collect()
}

/// Convert block indices to sample indices.
///
/// # Arguments
/// * `blocks` - Block index or array of block indices
/// * `block_length` - Number of frames per block
/// * `hop_length` - Number of samples between successive frames
///
/// # Returns
/// Sample indices corresponding to the beginning of each block
///
/// # Example
/// ```
/// use giggle::convert::blocks_to_samples;
///
/// let samples = blocks_to_samples(&[0, 1, 2], 16, 512);
/// assert_eq!(samples, vec![0, 8192, 16384]);
/// ```
pub fn blocks_to_samples(blocks: &[usize], block_length: usize, hop_length: usize) -> Vec<usize> {
    let frames = blocks_to_frames(blocks, block_length);
    frames_to_samples(&frames, hop_length)
}

/// Convert block indices to time (seconds).
///
/// # Arguments
/// * `blocks` - Block index or array of block indices
/// * `block_length` - Number of frames per block
/// * `hop_length` - Number of samples between successive frames
/// * `sr` - Sample rate
///
/// # Returns
/// Time values (in seconds) corresponding to the beginning of each block
///
/// # Example
/// ```
/// use giggle::convert::blocks_to_time;
///
/// let times = blocks_to_time(&[0, 1, 2], 16, 512, 22050);
/// assert!((times[1] - 0.372).abs() < 0.01);
/// ```
pub fn blocks_to_time(
    blocks: &[usize],
    block_length: usize,
    hop_length: usize,
    sr: u32,
) -> Vec<f32> {
    let samples = blocks_to_samples(blocks, block_length, hop_length);
    samples_to_time(&samples, sr)
}

/// Generate times similar to an input array shape.
///
/// # Arguments
/// * `n` - Number of time values to generate
/// * `sr` - Sample rate
/// * `hop_length` - Hop length
///
/// # Returns
/// Time values in seconds
///
/// # Example
/// ```
/// use giggle::convert::times_like;
///
/// let times = times_like(100, 22050, 512);
/// assert_eq!(times.len(), 100);
/// ```
pub fn times_like(n: usize, sr: u32, hop_length: usize) -> Vec<f32> {
    (0..n)
        .map(|i| i as f32 * hop_length as f32 / sr as f32)
        .collect()
}

/// Generate sample indices similar to an input array shape.
///
/// # Arguments
/// * `n` - Number of samples to generate
/// * `hop_length` - Hop length
///
/// # Returns
/// Sample indices
///
/// # Example
/// ```
/// use giggle::convert::samples_like;
///
/// let samples = samples_like(100, 512);
/// assert_eq!(samples.len(), 100);
/// assert_eq!(samples[1], 512);
/// ```
pub fn samples_like(n: usize, hop_length: usize) -> Vec<usize> {
    (0..n).map(|i| i * hop_length).collect()
}
