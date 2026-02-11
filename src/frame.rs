/// Compute frame indices for signal framing.
///
/// This function calculates the starting indices for framing a signal
/// into overlapping windows.
///
/// # Arguments
/// * `len` - Length of the input signal
/// * `frame_length` - Length of each frame
/// * `hop_length` - Number of samples to advance between frames
/// * `center` - If true, pad the signal to center frames (uses symmetric padding)
///
/// # Returns
/// Vector of starting indices for each frame
///
/// # Example
/// ```
/// use giggle::frame::frame_indices;
///
/// let indices = frame_indices(1000, 512, 256, true).unwrap();
/// assert_eq!(indices[0], 0);
/// assert_eq!(indices[1], 256);
/// ```
pub fn frame_indices(
    len: usize,
    frame_length: usize,
    hop_length: usize,
    center: bool,
) -> crate::Result<Vec<usize>> {
    if frame_length == 0 {
        return Err(crate::Error::InvalidSize {
            name: "frame_length",
            value: 0,
            reason: "must be > 0",
        });
    }
    if hop_length == 0 {
        return Err(crate::Error::InvalidSize {
            name: "hop_length",
            value: 0,
            reason: "must be > 0",
        });
    }
    let padded_len = if center { len + frame_length } else { len };
    if padded_len < frame_length {
        return Ok(Vec::new());
    }
    let n_frames = (padded_len - frame_length) / hop_length + 1;
    Ok((0..n_frames).map(|i| i * hop_length).collect())
}

/// Frame a signal into overlapping windows.
///
/// This function splits a signal into overlapping frames, optionally with
/// center padding (symmetric zero-padding at both ends).
///
/// # Arguments
/// * `y` - Input audio signal
/// * `frame_length` - Length of each frame
/// * `hop_length` - Number of samples to advance between frames
/// * `center` - If true, pad the signal to center frames
///
/// # Returns
/// Vector of frames, where each frame is a Vec<f32>
///
/// # Example
/// ```
/// use giggle::frame::frame_signal;
///
/// let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let frames = frame_signal(&signal, 4, 2, false).unwrap();
/// assert_eq!(frames.len(), 2); // 2 frames
/// assert_eq!(frames[0].len(), 4);
/// ```
pub fn frame_signal(
    y: &[f32],
    frame_length: usize,
    hop_length: usize,
    center: bool,
) -> crate::Result<Vec<Vec<f32>>> {
    if frame_length == 0 {
        return Err(crate::Error::InvalidSize {
            name: "frame_length",
            value: 0,
            reason: "must be > 0",
        });
    }
    if hop_length == 0 {
        return Err(crate::Error::InvalidSize {
            name: "hop_length",
            value: 0,
            reason: "must be > 0",
        });
    }

    let pad = if center { frame_length / 2 } else { 0 };
    let mut padded = vec![0.0f32; y.len() + 2 * pad];
    padded[pad..pad + y.len()].copy_from_slice(y);

    let indices = frame_indices(y.len(), frame_length, hop_length, center)?;
    let mut frames = Vec::new();
    for start in indices {
        let end = start + frame_length;
        let slice = &padded[start..end];
        frames.push(slice.to_vec());
    }

    Ok(frames)
}
