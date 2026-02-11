use super::*;
use approx::assert_relative_eq;
use ndarray::Array2;

#[test]
fn test_autocorrelate() {
    let signal = vec![1.0, 0.5, -0.5, -1.0, -0.5, 0.5, 1.0];
    let acf = autocorrelate(&signal, Some(4));

    assert_eq!(acf.len(), 4);
    // acf is unnormalized, so acf[0] is the energy
    assert!(acf[0] > 0.0);
    assert!(acf[0] >= acf[1].abs());
}

#[test]
fn test_autocorrelate_empty() {
    let signal = vec![];
    let acf = autocorrelate(&signal, None);
    assert_eq!(acf.len(), 0);
}

#[test]
fn test_valid_audio() {
    // Valid audio
    let y = vec![0.0, 0.5, -0.5, 0.0];
    assert!(valid_audio(&y).is_ok());

    // Empty audio
    let empty: Vec<f32> = vec![];
    assert!(valid_audio(&empty).is_err());

    // NaN in audio
    let with_nan = vec![0.0, f32::NAN, 0.5];
    assert!(valid_audio(&with_nan).is_err());

    // Inf in audio
    let with_inf = vec![0.0, f32::INFINITY, 0.5];
    assert!(valid_audio(&with_inf).is_err());

    // Negative infinity
    let with_neg_inf = vec![0.0, f32::NEG_INFINITY, 0.5];
    assert!(valid_audio(&with_neg_inf).is_err());
}

#[test]
fn test_valid_audio_2d() {
    use ndarray::Array2;

    // Valid 2D audio
    let y =
        Array2::from_shape_vec((2, 4), vec![0.0, 0.5, -0.5, 0.0, 0.1, 0.2, -0.1, -0.2]).unwrap();
    assert!(valid_audio_2d(&y).is_ok());

    // Empty 2D audio
    let empty = Array2::<f32>::zeros((0, 0));
    assert!(valid_audio_2d(&empty).is_err());

    // With NaN
    let with_nan = Array2::from_shape_vec((2, 2), vec![0.0, f32::NAN, 0.1, 0.2]).unwrap();
    assert!(valid_audio_2d(&with_nan).is_err());
}

#[test]
fn test_valid_int() {
    // Valid integers
    assert!(valid_int(5.0, "test").is_ok());
    assert!(valid_int(-3.0, "test").is_ok());
    assert!(valid_int(0.0, "test").is_ok());

    // Non-integer
    assert!(valid_int(5.5, "test").is_err());
    assert!(valid_int(-3.2, "test").is_err());

    // NaN
    assert!(valid_int(f32::NAN, "test").is_err());

    // Infinity
    assert!(valid_int(f32::INFINITY, "test").is_err());
    assert!(valid_int(f32::NEG_INFINITY, "test").is_err());
}

#[test]
fn test_is_positive_int() {
    // Positive integers
    assert!(is_positive_int(1.0));
    assert!(is_positive_int(5.0));
    assert!(is_positive_int(100.0));

    // Zero
    assert!(!is_positive_int(0.0));

    // Negative
    assert!(!is_positive_int(-1.0));
    assert!(!is_positive_int(-5.0));

    // Non-integer
    assert!(!is_positive_int(5.5));
    assert!(!is_positive_int(3.15));

    // NaN and Inf
    assert!(!is_positive_int(f32::NAN));
    assert!(!is_positive_int(f32::INFINITY));
    assert!(!is_positive_int(f32::NEG_INFINITY));
}

#[test]
fn test_valid_intervals() {
    // Valid intervals
    let intervals = vec![(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)];
    assert!(valid_intervals(&intervals, false).is_ok());
    assert!(valid_intervals(&intervals, true).is_ok());

    // Empty intervals
    let empty: Vec<(f32, f32)> = vec![];
    assert!(valid_intervals(&empty, false).is_ok());

    // Invalid: start >= end
    let invalid = vec![(0.0, 1.0), (2.0, 1.5)];
    assert!(valid_intervals(&invalid, false).is_err());

    // Overlapping intervals (strict mode)
    let overlapping = vec![(0.0, 2.0), (1.5, 3.0)];
    assert!(valid_intervals(&overlapping, false).is_ok());
    assert!(valid_intervals(&overlapping, true).is_err());

    // Non-finite boundaries
    let non_finite = vec![(0.0, f32::NAN)];
    assert!(valid_intervals(&non_finite, false).is_err());

    let non_finite_inf = vec![(f32::INFINITY, 1.0)];
    assert!(valid_intervals(&non_finite_inf, false).is_err());
}

#[test]
fn test_fix_frames() {
    // Basic clipping
    let frames = vec![0, 50, 100, 150, 200, 250];
    let fixed = fix_frames(&frames, Some(0), Some(150), true).unwrap();
    assert_eq!(fixed, vec![0, 50, 100, 150]);

    // Pad to span up to 250
    let frames = vec![0, 50, 100, 150, 200];
    let fixed = fix_frames(&frames, Some(0), Some(250), true).unwrap();
    assert_eq!(fixed, vec![0, 50, 100, 150, 200, 250]);

    // No pad
    let frames = vec![0, 50, 100, 150, 200];
    let fixed = fix_frames(&frames, Some(0), Some(250), false).unwrap();
    assert_eq!(fixed, vec![0, 50, 100, 150, 200]);

    // Starting away from zero
    let frames = vec![200, 233, 266, 299, 332];
    let fixed = fix_frames(&frames, Some(0), None, true).unwrap();
    assert_eq!(fixed, vec![0, 200, 233, 266, 299, 332]);

    // With x_max
    let frames = vec![200, 233, 266, 299, 332];
    let fixed = fix_frames(&frames, Some(0), Some(400), true).unwrap();
    assert_eq!(fixed, vec![0, 200, 233, 266, 299, 332, 400]);

    // Duplicates should be removed
    let frames = vec![0, 50, 50, 100, 100, 150];
    let fixed = fix_frames(&frames, Some(0), Some(150), true).unwrap();
    assert_eq!(fixed, vec![0, 50, 100, 150]);

    // Empty input
    let empty: Vec<usize> = vec![];
    let fixed = fix_frames(&empty, Some(0), Some(100), true).unwrap();
    assert_eq!(fixed, vec![0, 100]);
}

#[test]
fn test_fix_frames_f32() {
    let frames = vec![0.0, 50.0, 100.0, 150.0, 200.0];
    let fixed = fix_frames_f32(&frames, Some(0), Some(150), true).unwrap();
    assert_eq!(fixed, vec![0, 50, 100, 150]);

    // Negative values should be rejected
    let frames = vec![0.0, -50.0, 100.0];
    assert!(fix_frames_f32(&frames, Some(0), Some(100), true).is_err());
}

#[test]
fn test_buf_to_float_8bit() {
    // 8-bit samples (0-255, interpreted as signed -128 to 127)
    let bytes = vec![0u8, 127, 128, 255]; // 0, 127, -128, -1
    let float_samples = buf_to_float(&bytes, 1).unwrap();

    assert_relative_eq!(float_samples[0], 0.0, epsilon = 1e-6);
    assert_relative_eq!(float_samples[1], 127.0 / 128.0, epsilon = 1e-6);
    assert_relative_eq!(float_samples[2], -128.0 / 128.0, epsilon = 1e-6);
    assert_relative_eq!(float_samples[3], -1.0 / 128.0, epsilon = 1e-6);
}

#[test]
fn test_buf_to_float_16bit() {
    // 16-bit samples in little-endian
    let bytes = vec![
        0x00, 0x00, // 0
        0xFF, 0x7F, // 32767
        0x00, 0x80, // -32768
        0x00, 0x40, // 16384
    ];
    let float_samples = buf_to_float(&bytes, 2).unwrap();

    assert_relative_eq!(float_samples[0], 0.0, epsilon = 1e-6);
    assert_relative_eq!(float_samples[1], 32767.0 / 32768.0, epsilon = 1e-6);
    assert_relative_eq!(float_samples[2], -32768.0 / 32768.0, epsilon = 1e-6);
    assert_relative_eq!(float_samples[3], 16384.0 / 32768.0, epsilon = 1e-6);
}

#[test]
fn test_buf_to_float_32bit() {
    // 32-bit samples in little-endian
    let bytes = vec![
        0x00, 0x00, 0x00, 0x00, // 0
        0xFF, 0xFF, 0xFF, 0x7F, // 2147483647
        0x00, 0x00, 0x00, 0x80, // -2147483648
    ];
    let float_samples = buf_to_float(&bytes, 4).unwrap();

    assert_relative_eq!(float_samples[0], 0.0, epsilon = 1e-6);
    assert_relative_eq!(
        float_samples[1],
        2147483647.0 / 2147483648.0,
        epsilon = 1e-6
    );
    assert_relative_eq!(
        float_samples[2],
        -2147483648.0 / 2147483648.0,
        epsilon = 1e-6
    );
}

#[test]
fn test_buf_to_float_errors() {
    // Invalid n_bytes
    assert!(buf_to_float(&[0u8, 0u8], 3).is_err());
    assert!(buf_to_float(&[0u8, 0u8], 0).is_err());

    // Buffer length not multiple of n_bytes
    assert!(buf_to_float(&[0u8, 0u8, 0u8], 2).is_err());
}

#[test]
fn test_i16_to_float() {
    let samples = vec![0i16, 32767, -32768, 16384];
    let float_samples = i16_to_float(&samples);

    assert_relative_eq!(float_samples[0], 0.0, epsilon = 1e-6);
    assert_relative_eq!(float_samples[1], 32767.0 / 32768.0, epsilon = 1e-6);
    assert_relative_eq!(float_samples[2], -32768.0 / 32768.0, epsilon = 1e-6);
    assert_relative_eq!(float_samples[3], 16384.0 / 32768.0, epsilon = 1e-6);
}

#[test]
fn test_i32_to_float() {
    let samples = vec![0i32, 2147483647, -2147483648];
    let float_samples = i32_to_float(&samples);

    assert_relative_eq!(float_samples[0], 0.0, epsilon = 1e-6);
    assert_relative_eq!(
        float_samples[1],
        2147483647.0 / 2147483648.0,
        epsilon = 1e-6
    );
    assert_relative_eq!(
        float_samples[2],
        -2147483648.0 / 2147483648.0,
        epsilon = 1e-6
    );
}

#[test]
fn test_index_to_slice() {
    // Generate slices from spaced indices
    let idx = vec![20, 35, 50, 65, 80, 95];
    let slices = index_to_slice(&idx, None, None, None, true).unwrap();
    assert_eq!(slices.len(), 5);
    assert_eq!(slices[0], (20..35));
    assert_eq!(slices[1], (35..50));
    assert_eq!(slices[2], (50..65));
    assert_eq!(slices[3], (65..80));
    assert_eq!(slices[4], (80..95));
}

#[test]
fn test_index_to_slice_with_padding() {
    // Pad to span the range (0, 100)
    let idx = vec![20, 35, 50, 65, 80, 95];
    let slices = index_to_slice(&idx, Some(0), Some(100), None, true).unwrap();
    assert_eq!(slices.len(), 7);
    assert_eq!(slices[0], (0..20)); // Padded at start
    assert_eq!(slices[1], (20..35));
    assert_eq!(slices[2], (35..50));
    assert_eq!(slices[3], (50..65));
    assert_eq!(slices[4], (65..80));
    assert_eq!(slices[5], (80..95));
    assert_eq!(slices[6], (95..100)); // Padded at end
}

#[test]
fn test_index_to_slice_no_padding() {
    // No padding
    let idx = vec![20, 35, 50, 65, 80, 95];
    let slices = index_to_slice(&idx, Some(0), Some(100), None, false).unwrap();
    assert_eq!(slices.len(), 5);
    // Without padding, should just be the original indices
    assert_eq!(slices[0], (20..35));
    assert_eq!(slices[4], (80..95));
}

#[test]
fn test_index_to_slice_empty() {
    // Empty input with padding
    let idx: Vec<usize> = vec![];
    let slices = index_to_slice(&idx, Some(0), Some(100), None, true).unwrap();
    assert_eq!(slices.len(), 1);
    assert_eq!(slices[0], (0..100));
}

#[test]
fn test_index_to_slice_f32() {
    let idx = vec![20.0, 35.0, 50.0, 65.0];
    let slices = index_to_slice_f32(&idx, None, None, None, true).unwrap();
    assert_eq!(slices.len(), 3);
    assert_eq!(slices[0], (20..35));
    assert_eq!(slices[1], (35..50));
    assert_eq!(slices[2], (50..65));
}

#[test]
fn test_index_to_slice_f32_negative() {
    // Negative values should be rejected
    let idx = vec![20.0, -5.0, 50.0];
    assert!(index_to_slice_f32(&idx, None, None, None, true).is_err());
}

#[test]
fn test_index_to_slice_insufficient_indices() {
    // Single index should error
    let idx = vec![20];
    assert!(index_to_slice(&idx, None, None, None, true).is_err());
}

#[test]
fn test_normalize_l1() {
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let normalized = normalize(&x, NormType::L1);

    let sum: f32 = normalized.iter().map(|v| v.abs()).sum();
    assert_relative_eq!(sum, 1.0, epsilon = 0.01);
}

#[test]
fn test_normalize_l2() {
    let x = vec![3.0, 4.0];
    let normalized = normalize(&x, NormType::L2);

    let norm: f32 = normalized.iter().map(|v| v * v).sum::<f32>().sqrt();
    assert_relative_eq!(norm, 1.0, epsilon = 0.01);
}

#[test]
fn test_normalize_max() {
    let x = vec![1.0, 5.0, 3.0, -7.0];
    let normalized = normalize(&x, NormType::Max);

    let max = normalized.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    assert_relative_eq!(max, 1.0, epsilon = 0.01);
}

#[test]
fn test_normalize_2d() {
    let x = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

    let normalized = normalize_2d(&x, NormType::L2, 1);
    assert_eq!(normalized.shape(), x.shape());
}

#[test]
fn test_localmax() {
    let signal = vec![0.0, 1.0, 0.5, 2.0, 0.3, 1.5, 0.2];
    let peaks = localmax(&signal);

    assert!(peaks.contains(&1));
    assert!(peaks.contains(&3));
    assert!(peaks.contains(&5));
    assert_eq!(peaks.len(), 3);
}

#[test]
fn test_localmax_empty() {
    let signal = vec![1.0, 2.0];
    let peaks = localmax(&signal);
    assert_eq!(peaks.len(), 0);
}

#[test]
fn test_localmin_basic() {
    let x = vec![1.0, 0.0, 1.0, 2.0, -1.0, 0.0, -2.0, 1.0];
    let mins = localmin(&x);

    assert_eq!(mins.len(), 8);
    assert!(!mins[0]); // First element never a minimum
    assert!(mins[1]); // 0.0 < 1.0 and 0.0 <= 1.0
    assert!(!mins[2]);
    assert!(!mins[3]);
    assert!(mins[4]); // -1.0 < 2.0 and -1.0 <= 0.0
    assert!(!mins[5]);
    assert!(mins[6]); // -2.0 < 0.0 and -2.0 <= 1.0
}

#[test]
fn test_localmin_empty() {
    let x: Vec<f32> = Vec::new();
    let mins = localmin(&x);
    assert!(mins.is_empty());
}

#[test]
fn test_localmin_single() {
    let x = vec![1.0];
    let mins = localmin(&x);
    assert_eq!(mins, vec![false]);
}

#[test]
fn test_localmin_2d_axis0() {
    let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 0.0, 1.0, 1.0, 3.0]).unwrap();

    let mins = localmin_2d(&x, 0);
    assert_eq!(mins.shape(), &[3, 2]);
    // Column 0: values are 1, 0, 1 -> min at row 1
    assert!(mins[(1, 0)]);
    // Column 1: values are 2, 1, 3 -> min at row 1
    assert!(mins[(1, 1)]);
}

#[test]
fn test_localmin_2d_axis1() {
    let x = Array2::from_shape_vec((2, 3), vec![1.0, 0.0, 1.0, 2.0, 1.0, 3.0]).unwrap();

    let mins = localmin_2d(&x, 1);
    assert_eq!(mins.shape(), &[2, 3]);
    // Row 0: values are 1, 0, 1 -> min at col 1
    assert!(mins[(0, 1)]);
    // Row 1: values are 2, 1, 3 -> min at col 1
    assert!(mins[(1, 1)]);
}

#[test]
fn test_axis_sort_by_row() {
    let s = Array2::from_shape_vec(
        (3, 4),
        vec![
            0.0, 1.0, 2.0, 0.0, // peak at col 2
            1.0, 0.0, 0.0, 0.0, // peak at col 0
            0.0, 0.0, 0.0, 3.0, // peak at col 3
        ],
    )
    .unwrap();

    let (sorted, idx) = axis_sort(&s, 0, false);

    // Should sort rows by peak column: row1 (col 0), row0 (col 2), row2 (col 3)
    assert_eq!(idx, vec![1, 0, 2]);
    assert_eq!(sorted[(0, 0)], 1.0); // First row is now original row 1
}

#[test]
fn test_axis_sort_by_col() {
    let s =
        Array2::from_shape_vec((3, 3), vec![0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 0.0, 3.0]).unwrap();

    let (_sorted, idx) = axis_sort(&s, 1, false);

    // Sort columns by peak row
    // Col 0: peak at row 1, Col 1: peak at row 0, Col 2: peak at row 2
    assert_eq!(idx, vec![1, 0, 2]);
}

#[test]
fn test_axis_sort_by_argmin() {
    let s = Array2::from_shape_vec(
        (3, 3),
        vec![
            1.0, 2.0, 0.0, // min at col 2
            0.0, 1.0, 2.0, // min at col 0
            2.0, 0.0, 1.0, // min at col 1
        ],
    )
    .unwrap();

    let (_, idx) = axis_sort(&s, 0, true);

    // Sort by minimum column: row1 (col 0), row2 (col 1), row0 (col 2)
    assert_eq!(idx, vec![1, 2, 0]);
}

#[test]
fn test_axis_sort_empty() {
    let s = Array2::<f32>::zeros((0, 0));
    let (sorted, idx) = axis_sort(&s, 0, false);
    assert_eq!(sorted.shape(), &[0, 0]);
    assert!(idx.is_empty());
}

#[test]
fn test_sparsify_rows_basic() {
    let x = Array2::from_shape_vec((1, 8), vec![0.0, 0.1, 0.3, 0.5, 0.8, 0.5, 0.3, 0.1]).unwrap();

    let sparse = sparsify_rows(&x, 0.1);

    assert_eq!(sparse.n_rows, 1);
    assert_eq!(sparse.n_cols, 8);
    // Should have fewer non-zeros than 8
    assert!(sparse.nnz() < 8);

    // Convert back to dense and check
    let dense = sparse.to_dense();
    assert_eq!(dense.shape(), &[1, 8]);
    // Largest values should be preserved
    assert!(dense[(0, 4)] > 0.0); // 0.8 should be kept
}

#[test]
fn test_sparsify_rows_multi_row() {
    let x = Array2::from_shape_vec((2, 4), vec![0.1, 0.5, 0.3, 0.1, 0.2, 0.1, 0.6, 0.1]).unwrap();

    let sparse = sparsify_rows(&x, 0.2);

    // Each row processed independently
    let dense = sparse.to_dense();
    // Row 0 peak at col 1, Row 1 peak at col 2
    assert!(dense[(0, 1)] > 0.0);
    assert!(dense[(1, 2)] > 0.0);
}

#[test]
fn test_sparsify_rows_empty() {
    let x = Array2::<f32>::zeros((0, 0));
    let sparse = sparsify_rows(&x, 0.1);
    assert_eq!(sparse.nnz(), 0);
}

#[test]
fn test_sync_mean() {
    let data = Array2::from_shape_vec(
        (2, 10),
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0,
            3.0, 2.0, 1.0,
        ],
    )
    .unwrap();

    let beats = vec![0, 5, 10];
    let synced = sync(&data, &beats, "mean", true);

    assert_eq!(synced.shape(), &[2, 2]);
    // First segment [0,5): mean of 1,2,3,4,5 = 3.0
    assert_relative_eq!(synced[(0, 0)], 3.0, epsilon = 0.01);
    // Second segment [5,10): mean of 6,7,8,9,10 = 8.0
    assert_relative_eq!(synced[(0, 1)], 8.0, epsilon = 0.01);
}

#[test]
fn test_sync_median() {
    let data = Array2::from_shape_vec((1, 5), vec![1.0, 2.0, 3.0, 4.0, 100.0]).unwrap();

    let beats = vec![0, 5];
    let synced = sync(&data, &beats, "median", true);

    assert_eq!(synced.shape(), &[1, 1]);
    // Median of 1,2,3,4,100 = 3.0
    assert_relative_eq!(synced[(0, 0)], 3.0, epsilon = 0.01);
}

#[test]
fn test_sync_min_max() {
    let data = Array2::from_shape_vec((1, 4), vec![3.0, 1.0, 4.0, 2.0]).unwrap();

    let beats = vec![0, 4];

    let synced_min = sync(&data, &beats, "min", true);
    let synced_max = sync(&data, &beats, "max", true);

    assert_relative_eq!(synced_min[(0, 0)], 1.0, epsilon = 0.01);
    assert_relative_eq!(synced_max[(0, 0)], 4.0, epsilon = 0.01);
}

#[test]
fn test_sync_no_pad() {
    let data = Array2::from_shape_vec(
        (1, 10),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    )
    .unwrap();

    // Boundaries don't cover full range
    let beats = vec![2, 5, 8];
    let synced = sync(&data, &beats, "mean", false);

    // Should have 2 segments: [2,5) and [5,8)
    assert_eq!(synced.shape(), &[1, 2]);
}

#[test]
fn test_sync_empty() {
    let data = Array2::<f32>::zeros((2, 0));
    let beats = vec![0];
    let synced = sync(&data, &beats, "mean", true);
    assert_eq!(synced.shape(), &[2, 0]);
}

#[test]
fn test_peak_pick() {
    let signal = vec![0.0, 1.0, 0.0, 2.0, 0.0, 1.5, 0.0];
    let peaks = peak_pick(&signal, 1, 1, 1, 1, 0.5, 0);

    assert!(peaks.contains(&3));
    assert!(!peaks.is_empty());
}

#[test]
fn test_peak_pick_with_wait() {
    let signal = vec![0.0, 1.0, 0.0, 1.1, 0.0, 2.0, 0.0];
    let peaks = peak_pick(&signal, 1, 1, 1, 1, 0.3, 3);

    assert!(peaks.len() < 3);
}

#[test]
fn test_mse_existing() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![1.0, 2.0, 3.0];
    assert_relative_eq!(mse(&a, &b), 0.0, epsilon = 0.01);

    let c = vec![0.0, 0.0, 0.0];
    let error = mse(&a, &c);
    assert!(error > 0.0);
}

#[test]
fn test_pad_center() {
    let data = vec![1, 2, 3];
    let padded = pad_center(&data, 7, 0);
    assert_eq!(padded, vec![0, 0, 1, 2, 3, 0, 0]);

    let padded2 = pad_center(&data, 8, 0);
    assert_eq!(padded2, vec![0, 0, 1, 2, 3, 0, 0, 0]);
}

#[test]
fn test_pad_center_trim() {
    let data = vec![1, 2, 3, 4, 5];
    let trimmed = pad_center(&data, 3, 0);
    assert_eq!(trimmed, vec![2, 3, 4]);
}

#[test]
fn test_fix_length() {
    let data = vec![1.0, 2.0, 3.0];
    let fixed = fix_length(&data, 5, 0.0);
    assert_eq!(fixed, vec![1.0, 2.0, 3.0, 0.0, 0.0]);

    let trimmed = fix_length(&data, 2, 0.0);
    assert_eq!(trimmed, vec![1.0, 2.0]);
}

#[test]
fn test_expand_to() {
    assert_eq!(expand_to(100, None), 128);
    assert_eq!(expand_to(128, None), 128);
    assert_eq!(expand_to(129, None), 256);
    assert_eq!(expand_to(50, Some(100)), 128);
    assert_eq!(expand_to(0, None), 1);
}

#[test]
fn test_frame_count() {
    assert_eq!(frame_count(1000, 512, 256), 2);
    assert_eq!(frame_count(2048, 512, 512), 4);
    assert_eq!(frame_count(100, 512, 256), 0);
}

#[test]
fn test_frame_array() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let frames = frame_array(&data, 3, 2, 0.0, false);

    assert_eq!(frames.shape(), &[3, 2]);
    assert_eq!(frames[(0, 0)], 1.0);
    assert_eq!(frames[(1, 0)], 2.0);
    assert_eq!(frames[(2, 0)], 3.0);
    assert_eq!(frames[(0, 1)], 3.0);
    assert_eq!(frames[(1, 1)], 4.0);
    assert_eq!(frames[(2, 1)], 5.0);
}

#[test]
fn test_frame_array_centered() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let frames = frame_array(&data, 4, 2, 0.0, true);

    assert!(frames.shape()[1] > 0);
    // First frame should have padding at the start
    assert_eq!(frames[(0, 0)], 0.0);
    assert_eq!(frames[(1, 0)], 0.0);
}

#[test]
fn test_frame_array_empty() {
    let data: Vec<f32> = vec![];
    let frames = frame_array(&data, 3, 2, 0.0, false);
    assert_eq!(frames.shape(), &[3, 0]);
}

#[test]
fn test_tiny() {
    let val = 1.0f32;
    let t = tiny(&val);
    assert!(t > 0.0);
    assert!(t < 1e-6);
}

#[test]
fn test_softmask_basic() {
    let reference = Array2::from_shape_vec((2, 3), vec![2.0, 0.0, 4.0, 1.0, 3.0, 0.0]).unwrap();

    let other = Array2::from_shape_vec((2, 3), vec![2.0, 1.0, 0.0, 1.0, 1.0, 0.0]).unwrap();

    let mask = softmask(&reference, &[&other], 1.0, false);

    assert_eq!(mask.shape(), reference.shape());

    // Check specific values
    // (0,0): 2/(2+2) = 0.5
    assert_relative_eq!(mask[(0, 0)], 0.5, epsilon = 0.01);

    // (0,1): 0/(0+1) = 0.0
    assert_relative_eq!(mask[(0, 1)], 0.0, epsilon = 0.01);

    // (0,2): 4/(4+0) = 1.0
    assert_relative_eq!(mask[(0, 2)], 1.0, epsilon = 0.01);
}

#[test]
fn test_softmask_split_zeros() {
    let reference = Array2::from_shape_vec((2, 2), vec![0.0, 2.0, 0.0, 1.0]).unwrap();

    let other = Array2::from_shape_vec((2, 2), vec![0.0, 2.0, 0.0, 3.0]).unwrap();

    let mask = softmask(&reference, &[&other], 1.0, true);

    // (0,0): Both zero, split evenly: 1/2 = 0.5
    assert_relative_eq!(mask[(0, 0)], 0.5, epsilon = 0.01);

    // (1,0): Both zero, split evenly: 1/2 = 0.5
    assert_relative_eq!(mask[(1, 0)], 0.5, epsilon = 0.01);
}

#[test]
fn test_fill_off_diagonal() {
    let arr =
        Array2::from_shape_vec((3, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).unwrap();

    let filled = fill_off_diagonal(&arr, 0.0);

    // Diagonal should be unchanged
    assert_eq!(filled[(0, 0)], 1.0);
    assert_eq!(filled[(1, 1)], 5.0);
    assert_eq!(filled[(2, 2)], 9.0);

    // Off-diagonal should be 0
    assert_eq!(filled[(0, 1)], 0.0);
    assert_eq!(filled[(0, 2)], 0.0);
    assert_eq!(filled[(1, 0)], 0.0);
    assert_eq!(filled[(1, 2)], 0.0);
    assert_eq!(filled[(2, 0)], 0.0);
    assert_eq!(filled[(2, 1)], 0.0);
}

#[test]
fn test_shear_basic() {
    let arr = Array2::from_shape_vec(
        (3, 4),
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
    )
    .unwrap();

    // Shear along axis 0 with factor 1.0
    let sheared = shear(&arr, 1.0, 0);
    assert_eq!(sheared.shape(), arr.shape());

    // Check that shearing preserves shape
    assert_eq!(sheared.shape(), &[3, 4]);
}

#[test]
fn test_shear_zero_factor() {
    let arr = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

    // Zero factor should return unchanged array
    let sheared = shear(&arr, 0.0, 0);

    for i in 0..2 {
        for j in 0..3 {
            assert_eq!(sheared[(i, j)], arr[(i, j)]);
        }
    }
}

#[test]
fn test_cyclic_gradient() {
    let arr = Array2::from_shape_vec(
        (4, 4),
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ],
    )
    .unwrap();

    // Gradient along axis 0 (rows)
    let grad = cyclic_gradient(&arr, 0);
    assert_eq!(grad.shape(), arr.shape());

    // Gradient should capture differences
    // For uniform gradient, values should be consistent
    assert!(grad[(1, 0)] > 0.0); // Should be positive (increasing downward)
}

#[test]
fn test_cyclic_gradient_constant() {
    // Constant array should have zero gradient
    let arr = Array2::from_elem((3, 3), 5.0);
    let grad = cyclic_gradient(&arr, 0);

    for val in grad.iter() {
        assert_relative_eq!(*val, 0.0, epsilon = 1e-6);
    }
}

#[test]
fn test_stack_axis0() {
    let arr1 = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

    let arr2 = Array2::from_shape_vec((2, 3), vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();

    let stacked = stack(&[&arr1, &arr2], 0).unwrap();

    // Should have shape (4, 3) = (2*2, 3)
    assert_eq!(stacked.shape(), &[4, 3]);

    // Check values
    assert_eq!(stacked[(0, 0)], 1.0);
    assert_eq!(stacked[(2, 0)], 7.0); // Start of arr2
}

#[test]
fn test_stack_axis1() {
    let arr1 = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();

    let arr2 = Array2::from_shape_vec((2, 2), vec![5.0, 6.0, 7.0, 8.0]).unwrap();

    let stacked = stack(&[&arr1, &arr2], 1).unwrap();

    // Should have shape (2, 4) = (2, 2*2)
    assert_eq!(stacked.shape(), &[2, 4]);

    // Check values
    assert_eq!(stacked[(0, 0)], 1.0);
    assert_eq!(stacked[(0, 2)], 5.0); // Start of arr2
}

#[test]
fn test_stack_empty() {
    let arrays: Vec<&Array2<f32>> = vec![];
    let stacked = stack(&arrays, 0).unwrap();
    assert_eq!(stacked.shape(), &[0, 0]);
}

#[test]
fn test_count_unique() {
    let arr = vec![1.0, 2.0, 3.0, 2.0, 1.0, 4.0];
    assert_eq!(count_unique(&arr), 4); // 1, 2, 3, 4

    let arr2 = vec![5.0, 5.0, 5.0];
    assert_eq!(count_unique(&arr2), 1); // Only 5

    let arr3: Vec<f32> = vec![];
    assert_eq!(count_unique(&arr3), 0); // Empty
}

#[test]
fn test_is_unique() {
    let unique = vec![1.0, 2.0, 3.0, 4.0];
    assert!(is_unique(&unique));

    let not_unique = vec![1.0, 2.0, 1.0];
    assert!(!is_unique(&not_unique));

    let empty: Vec<f32> = vec![];
    assert!(is_unique(&empty)); // Empty is trivially unique
}

#[test]
fn test_abs2() {
    use num_complex::Complex32;

    let z = Complex32::new(3.0, 4.0);
    assert_relative_eq!(abs2(z), 25.0, epsilon = 1e-6); // 3^2 + 4^2 = 25

    let z2 = Complex32::new(1.0, 0.0);
    assert_relative_eq!(abs2(z2), 1.0, epsilon = 1e-6);

    let z3 = Complex32::new(0.0, 1.0);
    assert_relative_eq!(abs2(z3), 1.0, epsilon = 1e-6);
}

#[test]
fn test_phasor() {
    use std::f32::consts::PI;

    // Test at key angles
    let p0 = phasor(0.0);
    assert_relative_eq!(p0.re, 1.0, epsilon = 1e-6);
    assert_relative_eq!(p0.im, 0.0, epsilon = 1e-6);

    let p_pi_2 = phasor(PI / 2.0);
    assert_relative_eq!(p_pi_2.re, 0.0, epsilon = 1e-6);
    assert_relative_eq!(p_pi_2.im, 1.0, epsilon = 1e-6);

    let p_pi = phasor(PI);
    assert_relative_eq!(p_pi.re, -1.0, epsilon = 1e-6);
    assert_relative_eq!(p_pi.im, 0.0, epsilon = 1e-6);

    // Verify unit magnitude
    let p_any = phasor(1.234);
    assert_relative_eq!(abs2(p_any), 1.0, epsilon = 1e-6);
}

#[test]
fn test_nnls_basic() {
    // Simple 2x2 problem
    let a = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
    let b = vec![1.0, 2.0];
    let x = nnls(&a, &b, None);

    assert_eq!(x.len(), 2);
    assert_relative_eq!(x[0], 1.0, epsilon = 0.01);
    assert_relative_eq!(x[1], 2.0, epsilon = 0.01);
}

#[test]
fn test_nnls_non_negative() {
    // Problem that requires non-negativity constraint
    let a = Array2::from_shape_vec((3, 2), vec![1.0, 1.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
    let b = vec![1.0, 0.5, 0.0];
    let x = nnls(&a, &b, None);

    // All components should be non-negative
    for &v in &x {
        assert!(v >= -1e-6, "Solution should be non-negative: {}", v);
    }
}

#[test]
fn test_nnls_overdetermined() {
    // Overdetermined system (more equations than unknowns)
    let a = Array2::from_shape_vec((4, 2), vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 3.0]).unwrap();
    let b = vec![1.0, 2.0, 3.0, 5.0];
    let x = nnls(&a, &b, None);

    assert_eq!(x.len(), 2);

    // Check non-negativity
    for &v in &x {
        assert!(v >= -1e-6);
    }

    // Check that it minimizes residual
    let mut residual = [0.0; 4];
    for i in 0..4 {
        let mut sum = 0.0;
        for j in 0..2 {
            sum += a[(i, j)] * x[j];
        }
        residual[i] = b[i] - sum;
    }

    let residual_norm: f32 = residual.iter().map(|r| r * r).sum::<f32>().sqrt();
    assert!(residual_norm < 10.0); // Should have reasonable residual
}

#[test]
fn test_nnls_empty() {
    let a = Array2::from_shape_vec((0, 0), vec![]).unwrap();
    let b = vec![];
    let x = nnls(&a, &b, None);

    assert_eq!(x.len(), 0);
}

#[test]
fn test_nnls_zero_solution() {
    // Problem where optimal solution is zero
    let a = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
    let b = vec![-1.0, -2.0]; // Negative target
    let x = nnls(&a, &b, None);

    // Should return zero (or very small) values
    for &v in &x {
        assert!(v >= -1e-6);
        assert!(v < 1.0); // Should be small
    }
}

#[test]
fn test_match_intervals_exact() {
    // Perfect overlap
    let reference = vec![(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)];
    let estimated = vec![(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)];
    let matches = match_intervals(&reference, &estimated, 0.5);

    assert_eq!(matches.len(), 3);
    assert_eq!(matches[0], (0, 0));
    assert_eq!(matches[1], (1, 1));
    assert_eq!(matches[2], (2, 2));
}

#[test]
fn test_match_intervals_partial() {
    // Partial overlap
    let reference = vec![(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)];
    let estimated = vec![(0.1, 1.1), (1.9, 3.1)];
    let matches = match_intervals(&reference, &estimated, 0.3);

    assert!(matches.len() >= 2, "Should match at least 2 intervals");

    // Check that matches are reasonable
    for &(ref_idx, est_idx) in &matches {
        assert!(ref_idx < reference.len());
        assert!(est_idx < estimated.len());
    }
}

#[test]
fn test_match_intervals_no_overlap() {
    // No overlap
    let reference = vec![(0.0, 1.0), (2.0, 3.0)];
    let estimated = vec![(5.0, 6.0), (7.0, 8.0)];
    let matches = match_intervals(&reference, &estimated, 0.5);

    assert_eq!(matches.len(), 0, "No intervals should match");
}

#[test]
fn test_match_intervals_empty() {
    let reference = vec![];
    let estimated = vec![(0.0, 1.0)];
    let matches = match_intervals(&reference, &estimated, 0.5);

    assert_eq!(matches.len(), 0);

    let reference = vec![(0.0, 1.0)];
    let estimated = vec![];
    let matches = match_intervals(&reference, &estimated, 0.5);

    assert_eq!(matches.len(), 0);
}

#[test]
fn test_match_intervals_invalid() {
    // Invalid intervals (end < start) should be skipped
    let reference = vec![(1.0, 0.0), (0.0, 1.0)];
    let estimated = vec![(0.0, 1.0)];
    let matches = match_intervals(&reference, &estimated, 0.5);

    // Only the valid interval should match
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].0, 1); // Second reference interval
}

#[test]
fn test_match_events_exact() {
    // Exact matches
    let reference = vec![0.5, 1.0, 1.5, 2.0];
    let estimated = vec![0.5, 1.0, 1.5, 2.0];
    let matches = match_events(&reference, &estimated, 0.1);

    assert_eq!(matches.len(), 4);
    for (i, &m) in matches.iter().enumerate().take(4) {
        assert_eq!(m, (i, i));
    }
}

#[test]
fn test_match_events_near() {
    // Near matches within window
    let reference = vec![0.5, 1.0, 1.5, 2.0];
    let estimated = vec![0.51, 1.02, 1.48];
    let matches = match_events(&reference, &estimated, 0.1);

    assert_eq!(matches.len(), 3);
    assert_eq!(matches[0], (0, 0)); // 0.5 -> 0.51
    assert_eq!(matches[1], (1, 1)); // 1.0 -> 1.02
    assert_eq!(matches[2], (2, 2)); // 1.5 -> 1.48
}

#[test]
fn test_match_events_outside_window() {
    // Events outside window should not match
    let reference = vec![0.0, 1.0, 2.0];
    let estimated = vec![0.5, 1.5];
    let matches = match_events(&reference, &estimated, 0.1);

    // No matches because all events are too far apart
    assert_eq!(matches.len(), 0);
}

#[test]
fn test_match_events_one_to_one() {
    // Each estimated event should only match once
    let reference = vec![1.0, 1.01, 1.02];
    let estimated = vec![1.0];
    let matches = match_events(&reference, &estimated, 0.1);

    // Only the closest reference event should match
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0], (0, 0)); // First ref event is closest
}

#[test]
fn test_match_events_empty() {
    let reference = vec![];
    let estimated = vec![1.0];
    let matches = match_events(&reference, &estimated, 0.1);

    assert_eq!(matches.len(), 0);

    let reference = vec![1.0];
    let estimated = vec![];
    let matches = match_events(&reference, &estimated, 0.1);

    assert_eq!(matches.len(), 0);
}

#[test]
fn test_cross_similarity_cosine() {
    let ref_feat = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]).unwrap();
    let est_feat = Array2::from_shape_vec((2, 2), vec![1.0, 3.0, 1.0, 3.0]).unwrap();

    let sim = cross_similarity(&ref_feat, &est_feat, "cosine");

    assert_eq!(sim.shape(), &[3, 2]);

    // Self-similarity should be 1.0
    assert!(
        (sim[(0, 0)] - 1.0).abs() < 0.01,
        "Identical vectors should have similarity 1.0"
    );
    assert!((sim[(2, 1)] - 1.0).abs() < 0.01);

    // All values should be in [-1, 1]
    for val in sim.iter() {
        assert!(
            val.abs() <= 1.01,
            "Cosine similarity should be in [-1,1], got {}",
            val
        );
    }
}

#[test]
fn test_cross_similarity_euclidean() {
    let ref_feat = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 1.0, 2.0]).unwrap();
    let est_feat = Array2::from_shape_vec((2, 2), vec![1.0, 5.0, 1.0, 5.0]).unwrap();

    let sim = cross_similarity(&ref_feat, &est_feat, "euclidean");

    assert_eq!(sim.shape(), &[2, 2]);

    // Identical vectors have distance 0 (similarity 0)
    assert!((sim[(0, 0)] - 0.0).abs() < 0.01);

    // Different vectors have negative similarity
    assert!(sim[(0, 1)] < 0.0);
}

#[test]
fn test_cross_similarity_manhattan() {
    let ref_feat = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 1.0, 2.0]).unwrap();
    let est_feat = Array2::from_shape_vec((2, 2), vec![1.0, 3.0, 1.0, 3.0]).unwrap();

    let sim = cross_similarity(&ref_feat, &est_feat, "manhattan");

    assert_eq!(sim.shape(), &[2, 2]);

    // All values should be non-positive (negative distance)
    for val in sim.iter() {
        assert!(*val <= 0.01);
    }
}

#[test]
fn test_cross_similarity_shape_mismatch() {
    let ref_feat = Array2::from_shape_vec((3, 2), vec![1.0; 6]).unwrap();
    let est_feat = Array2::from_shape_vec((2, 2), vec![1.0; 4]).unwrap();

    let sim = cross_similarity(&ref_feat, &est_feat, "cosine");

    // Should return zeros when feature dimensions don't match
    assert_eq!(sim.shape(), &[2, 2]);
    for val in sim.iter() {
        assert_eq!(*val, 0.0);
    }
}

#[test]
fn test_cross_similarity_empty() {
    let ref_feat = Array2::<f32>::zeros((2, 0));
    let est_feat = Array2::<f32>::zeros((2, 0));

    let sim = cross_similarity(&ref_feat, &est_feat, "cosine");

    assert_eq!(sim.shape(), &[0, 0]);
}

#[test]
fn test_cross_similarity_unknown_metric() {
    let ref_feat = Array2::from_shape_vec((2, 2), vec![1.0; 4]).unwrap();
    let est_feat = Array2::from_shape_vec((2, 2), vec![1.0; 4]).unwrap();

    let sim = cross_similarity(&ref_feat, &est_feat, "unknown");

    // Should return zeros for unknown metric
    assert_eq!(sim.shape(), &[2, 2]);
    for val in sim.iter() {
        assert_eq!(*val, 0.0);
    }
}

#[test]
fn test_recurrence_matrix_affinity() {
    let features =
        Array2::from_shape_vec((2, 4), vec![1.0, 2.0, 1.0, 3.0, 1.0, 2.0, 1.0, 3.0]).unwrap();

    let rec = recurrence_matrix(&features, "affinity", "cosine", 0.0);

    assert_eq!(rec.shape(), &[4, 4]);

    // Diagonal should be 1.0 (self-similarity)
    for i in 0..4 {
        assert!((rec[(i, i)] - 1.0).abs() < 0.01);
    }

    // Frames 0 and 2 are identical, should have similarity 1.0
    assert!((rec[(0, 2)] - 1.0).abs() < 0.01);
    assert!((rec[(2, 0)] - 1.0).abs() < 0.01);
}

#[test]
fn test_recurrence_matrix_connectivity() {
    let features =
        Array2::from_shape_vec((2, 4), vec![1.0, 2.0, 1.0, 3.0, 1.0, 2.0, 1.0, 3.0]).unwrap();

    let rec = recurrence_matrix(&features, "connectivity", "cosine", 0.9);

    assert_eq!(rec.shape(), &[4, 4]);

    // Should be binary
    for val in rec.iter() {
        assert!(*val == 0.0 || *val == 1.0);
    }

    // Frames 0 and 2 are identical, should be connected
    assert_eq!(rec[(0, 2)], 1.0);
    assert_eq!(rec[(2, 0)], 1.0);
}

#[test]
fn test_recurrence_to_lag() {
    let mut rec = Array2::zeros((4, 4));
    for i in 0..4 {
        for j in 0..4 {
            rec[(i, j)] = ((i as i32 - j as i32).abs()) as f32;
        }
    }

    let lag = recurrence_to_lag(&rec, true, 1);

    assert_eq!(lag.shape(), &[4, 4]);

    // Lag 0 (diagonal) should all be 0
    for i in 0..4 {
        assert_eq!(lag[(i, 0)], 0.0);
    }

    // Lag 1 should all be 1
    for i in 0..3 {
        assert_eq!(lag[(i, 1)], 1.0);
    }
}

#[test]
fn test_lag_to_recurrence() {
    let lag = Array2::from_shape_vec(
        (4, 4),
        vec![
            1.0, 0.5, 0.2, 0.1, 1.0, 0.5, 0.2, 0.0, 1.0, 0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
        ],
    )
    .unwrap();

    let rec = lag_to_recurrence(&lag, 1);

    assert_eq!(rec.shape(), &[4, 4]);

    // Should be symmetric
    for i in 0..4 {
        for j in 0..4 {
            assert_eq!(rec[(i, j)], rec[(j, i)]);
        }
    }

    // Diagonal (lag 0) should be 1.0
    for i in 0..4 {
        assert_eq!(rec[(i, i)], 1.0);
    }
}

#[test]
fn test_recurrence_lag_roundtrip() {
    let mut rec_original = Array2::zeros((5, 5));
    for i in 0..5 {
        for j in 0..5 {
            rec_original[(i, j)] = 1.0 / (1.0 + ((i as i32 - j as i32).abs() as f32));
        }
    }

    let lag = recurrence_to_lag(&rec_original, true, 1);
    let rec_reconstructed = lag_to_recurrence(&lag, 1);

    assert_eq!(rec_reconstructed.shape(), rec_original.shape());

    // Check that reconstruction is close to original
    for i in 0..5 {
        for j in 0..5 {
            assert!((rec_reconstructed[(i, j)] - rec_original[(i, j)]).abs() < 0.01);
        }
    }
}

#[test]
fn test_recurrence_empty() {
    let features = Array2::<f32>::zeros((2, 0));
    let rec = recurrence_matrix(&features, "affinity", "cosine", 0.0);

    assert_eq!(rec.shape(), &[0, 0]);
}

#[test]
fn test_dtw_identical() {
    // Identical sequences should have zero distance
    let x = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]).unwrap();

    let (distance, path) = dtw(&x, &x, "euclidean");

    assert!(
        distance < 0.01,
        "Identical sequences should have distance ~0, got {}",
        distance
    );
    assert_eq!(path.len(), 3);
    assert_eq!(path[0], (0, 0));
    assert_eq!(path[2], (2, 2));
}

#[test]
fn test_dtw_stretched() {
    // Test alignment with time stretching
    let x = Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap();
    let y = Array2::from_shape_vec((1, 5), vec![1.0, 1.5, 2.0, 2.5, 3.0]).unwrap();

    let (distance, path) = dtw(&x, &y, "euclidean");

    assert!(distance.is_finite());
    assert!(!path.is_empty());
    // Path should start at (0,0) and end at (2,4)
    assert_eq!(path[0], (0, 0));
    assert_eq!(path[path.len() - 1], (2, 4));
}

#[test]
fn test_dtw_different_lengths() {
    let x = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 1.0, 2.0]).unwrap();
    let y = Array2::from_shape_vec((2, 4), vec![1.0, 1.5, 2.0, 2.5, 1.0, 1.5, 2.0, 2.5]).unwrap();

    let (distance, path) = dtw(&x, &y, "euclidean");

    assert!(distance.is_finite());
    assert!(path.len() >= 4); // At least as long as the longer sequence
}

#[test]
fn test_dtw_metrics() {
    let x = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]).unwrap();
    let y = Array2::from_shape_vec((2, 3), vec![1.1, 2.1, 3.1, 1.1, 2.1, 3.1]).unwrap();

    let (dist_euc, _) = dtw(&x, &y, "euclidean");
    let (dist_man, _) = dtw(&x, &y, "manhattan");
    let (dist_cos, _) = dtw(&x, &y, "cosine");

    // All distances should be small for similar sequences
    assert!(dist_euc < 1.0);
    assert!(dist_man < 1.0);
    assert!(dist_cos < 0.1);
}

#[test]
fn test_dtw_empty() {
    let x = Array2::<f32>::zeros((2, 0));
    let y = Array2::<f32>::zeros((2, 1));

    let (distance, path) = dtw(&x, &y, "euclidean");

    assert!(distance.is_infinite());
    assert_eq!(path.len(), 0);
}

#[test]
fn test_dtw_feature_mismatch() {
    let x = Array2::from_shape_vec((2, 3), vec![1.0; 6]).unwrap();
    let y = Array2::from_shape_vec((3, 3), vec![1.0; 9]).unwrap();

    let (distance, path) = dtw(&x, &y, "euclidean");

    assert!(distance.is_infinite());
    assert_eq!(path.len(), 0);
}

#[test]
fn test_dtw_distance_only() {
    let x = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]).unwrap();
    let y = Array2::from_shape_vec((2, 3), vec![1.1, 2.1, 3.1, 1.1, 2.1, 3.1]).unwrap();

    let dist1 = dtw_distance(&x, &y, "euclidean");
    let (dist2, _) = dtw(&x, &y, "euclidean");

    assert_eq!(dist1, dist2);
}

#[test]
fn test_dtw_path_monotonic() {
    let x = Array2::from_shape_vec((1, 4), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let y = Array2::from_shape_vec((1, 4), vec![1.0, 2.0, 3.0, 4.0]).unwrap();

    let (_, path) = dtw(&x, &y, "euclidean");

    // Path should be monotonically increasing
    for i in 1..path.len() {
        assert!(path[i].0 >= path[i - 1].0, "Path should be monotonic in x");
        assert!(path[i].1 >= path[i - 1].1, "Path should be monotonic in y");
    }
}

#[test]
fn test_dtw_with_steps() {
    let x = Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap();
    let y = Array2::from_shape_vec((1, 4), vec![1.0, 2.0, 2.5, 3.0]).unwrap();

    let (distance, steps) = dtw_with_steps(&x, &y, "euclidean");

    assert!(distance >= 0.0);
    assert_eq!(steps.shape(), &[3, 4]);

    // Steps should be valid indices (0, 1, or 2 for default steps)
    for &step in steps.iter() {
        assert!(step <= 2);
    }
}

#[test]
fn test_dtw_backtracking_basic() {
    let x = Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap();
    let y = Array2::from_shape_vec((1, 4), vec![1.0, 2.0, 2.5, 3.0]).unwrap();

    let (_distance, steps) = dtw_with_steps(&x, &y, "euclidean");
    let path = dtw_backtracking(&steps, None, false, None);

    assert!(!path.is_empty());
    // Path should start at (0, 0) and end at (n_x-1, n_y-1)
    assert_eq!(path[0], (0, 0));
    assert_eq!(path[path.len() - 1], (2, 3));

    // Path should be monotonically increasing
    for i in 1..path.len() {
        assert!(path[i].0 >= path[i - 1].0);
        assert!(path[i].1 >= path[i - 1].1);
    }
}

#[test]
fn test_dtw_backtracking_matches_dtw() {
    let x = Array2::from_shape_vec(
        (2, 5),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0],
    )
    .unwrap();
    let y = Array2::from_shape_vec(
        (2, 6),
        vec![1.0, 2.0, 2.5, 3.5, 4.5, 5.0, 1.0, 2.0, 2.5, 3.5, 4.5, 5.0],
    )
    .unwrap();

    let (dist1, path1) = dtw(&x, &y, "euclidean");
    let (dist2, steps) = dtw_with_steps(&x, &y, "euclidean");
    let path2 = dtw_backtracking(&steps, None, false, None);

    // Distances should match
    assert!((dist1 - dist2).abs() < 1e-5);

    // Paths should have same length and similar structure
    assert_eq!(path1.len(), path2.len());
    assert_eq!(path1[0], path2[0]);
    assert_eq!(path1[path1.len() - 1], path2[path2.len() - 1]);
}

#[test]
fn test_dtw_backtracking_subseq() {
    let x = Array2::from_shape_vec((1, 4), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let y = Array2::from_shape_vec((1, 6), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();

    let (_, steps) = dtw_with_steps(&x, &y, "euclidean");

    // Subsequence backtracking starting from column 4
    let path = dtw_backtracking(&steps, None, true, Some(4));

    assert!(!path.is_empty());
    // Should end at row 3 (n_x - 1)
    assert_eq!(path[path.len() - 1].0, 3);
    // Should end at column 4 (start)
    assert_eq!(path[path.len() - 1].1, 4);
}

#[test]
fn test_dtw_backtracking_empty() {
    let steps = Array2::<usize>::zeros((0, 0));
    let path = dtw_backtracking(&steps, None, false, None);
    assert!(path.is_empty());
}

#[test]
fn test_dtw_backtracking_invalid_start() {
    let steps = Array2::<usize>::zeros((3, 4));
    // start without subseq should return empty
    let path = dtw_backtracking(&steps, None, false, Some(2));
    assert!(path.is_empty());
}

#[test]
fn test_dtw_backtracking_custom_steps() {
    let x = Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap();
    let y = Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap();

    let (_, steps) = dtw_with_steps(&x, &y, "euclidean");

    // Custom step sizes (not used in this simple test, but function should accept them)
    let custom_steps = [(2, 2), (1, 2), (2, 1)];
    let path = dtw_backtracking(&steps, Some(&custom_steps), false, None);

    assert!(!path.is_empty());
    assert_eq!(path[0], (0, 0));
    assert_eq!(path[path.len() - 1], (2, 2));
}

#[test]
fn test_viterbi_simple() {
    // Clear preference: state 0 then state 1
    let prob = Array2::from_shape_vec(
        (2, 4),
        vec![
            0.9, 0.8, 0.2, 0.1, // State 0 probabilities
            0.1, 0.2, 0.8, 0.9, // State 1 probabilities
        ],
    )
    .unwrap();

    let states = viterbi(&prob, None, None);

    assert_eq!(states.len(), 4);
    // Should prefer state 0 early, state 1 late
    assert_eq!(states[0], 0);
    assert_eq!(states[3], 1);
}

#[test]
fn test_viterbi_with_transition() {
    let prob =
        Array2::from_shape_vec((2, 4), vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]).unwrap();

    // Strong self-loop (stay in same state)
    let trans = transition_loop(2, 0.9);

    // Initial state 0
    let p_init = vec![1.0, 0.0];

    let states = viterbi(&prob, Some(&trans), Some(&p_init));

    assert_eq!(states.len(), 4);
    // With strong loop and uniform observations, should stay in state 0
    assert_eq!(states[0], 0);
    // Most likely stays in state 0 due to transition bias
    assert!(states.iter().filter(|&&s| s == 0).count() >= 3);
}

#[test]
fn test_viterbi_empty() {
    let prob = Array2::<f32>::zeros((2, 0));
    let states = viterbi(&prob, None, None);
    assert_eq!(states.len(), 0);

    let prob = Array2::<f32>::zeros((0, 4));
    let states = viterbi(&prob, None, None);
    assert_eq!(states.len(), 0);
}

#[test]
fn test_viterbi_single_state() {
    let prob = Array2::from_shape_vec((1, 5), vec![0.5, 0.6, 0.7, 0.8, 0.9]).unwrap();
    let states = viterbi(&prob, None, None);

    assert_eq!(states.len(), 5);
    // Only one state, all should be 0
    for &s in &states {
        assert_eq!(s, 0);
    }
}

#[test]
fn test_viterbi_discriminative_basic() {
    // Discriminative probabilities: columns sum to 1
    let prob = Array2::from_shape_vec(
        (2, 5),
        vec![
            0.8, 0.6, 0.3, 0.2, 0.7, // P(state=0 | obs)
            0.2, 0.4, 0.7, 0.8, 0.3, // P(state=1 | obs)
        ],
    )
    .unwrap();

    // Transition matrix: prefer staying in same state
    let trans = Array2::from_shape_vec((2, 2), vec![0.9, 0.1, 0.1, 0.9]).unwrap();

    let (states, logp) = viterbi_discriminative(&prob, &trans, None, None);

    assert_eq!(states.len(), 5);
    assert!(logp.is_finite());

    // First state should be 0 (high prob), last should be 0 (high prob)
    assert_eq!(states[0], 0);
    assert_eq!(states[4], 0);
}

#[test]
fn test_viterbi_discriminative_with_marginal() {
    // Discriminative probabilities
    let prob =
        Array2::from_shape_vec((2, 4), vec![0.7, 0.6, 0.4, 0.3, 0.3, 0.4, 0.6, 0.7]).unwrap();

    let trans = Array2::from_shape_vec((2, 2), vec![0.8, 0.2, 0.2, 0.8]).unwrap();

    // State 0 is more common in training data
    let p_state = [0.7, 0.3];

    let (states, _) = viterbi_discriminative(&prob, &trans, Some(&p_state), None);
    assert_eq!(states.len(), 4);
}

#[test]
fn test_viterbi_discriminative_empty() {
    let prob = Array2::<f32>::zeros((0, 0));
    let trans = Array2::<f32>::zeros((0, 0));

    let (states, logp) = viterbi_discriminative(&prob, &trans, None, None);
    assert!(states.is_empty());
    assert!(logp == f32::NEG_INFINITY);
}

#[test]
fn test_viterbi_discriminative_uniform() {
    // Equal probabilities everywhere
    let prob = Array2::from_elem((3, 5), 1.0 / 3.0);

    let trans = transition_uniform(3);

    let (states, _) = viterbi_discriminative(&prob, &trans, None, None);
    assert_eq!(states.len(), 5);
}

#[test]
fn test_viterbi_binary_basic() {
    // Binary state probabilities (probability of state being active)
    let prob = Array2::from_shape_vec(
        (1, 10),
        vec![0.1, 0.7, 0.4, 0.3, 0.8, 0.9, 0.8, 0.2, 0.6, 0.3],
    )
    .unwrap();

    // 2x2 transition: inactive <-> active
    let trans = Array2::from_shape_vec(
        (2, 2),
        vec![
            0.9, 0.1, // from inactive
            0.3, 0.7, // from active
        ],
    )
    .unwrap();

    let (states, logp) = viterbi_binary(&prob, &trans, None, None);

    assert_eq!(states.shape(), &[1, 10]);
    assert_eq!(logp.len(), 1);

    // The strong sequence [0.8, 0.9, 0.8] should trigger active states
    // Checking that middle region has some 1s
    let active_count: usize = (4..7).filter(|&t| states[(0, t)] == 1).count();
    assert!(
        active_count >= 1,
        "Should have some active states in high-prob region"
    );
}

#[test]
fn test_viterbi_binary_multi_label() {
    // Two independent binary labels
    let prob = Array2::from_shape_vec(
        (2, 5),
        vec![
            0.9, 0.8, 0.2, 0.1, 0.1, // Label 0: starts high, ends low
            0.1, 0.2, 0.8, 0.9, 0.9, // Label 1: starts low, ends high
        ],
    )
    .unwrap();

    let trans = Array2::from_shape_vec((2, 2), vec![0.8, 0.2, 0.2, 0.8]).unwrap();

    let (states, logp) = viterbi_binary(&prob, &trans, None, None);

    assert_eq!(states.shape(), &[2, 5]);
    assert_eq!(logp.len(), 2);

    // Label 0 should be active initially
    assert_eq!(states[(0, 0)], 1);
    // Label 1 should be active at the end
    assert_eq!(states[(1, 4)], 1);
}

#[test]
fn test_viterbi_binary_with_marginal() {
    // Test with high probability of being active
    let prob = Array2::from_shape_vec((1, 6), vec![0.9, 0.9, 0.9, 0.9, 0.9, 0.9]).unwrap();

    let trans = Array2::from_shape_vec((2, 2), vec![0.9, 0.1, 0.1, 0.9]).unwrap();

    // High marginal and initial probability
    let p_state = [0.8];
    let p_init = [0.9];

    let (states, _) = viterbi_binary(&prob, &trans, Some(&p_state), Some(&p_init));

    assert_eq!(states.shape(), &[1, 6]);
    // With high probabilities everywhere, should be mostly active
    let active_count: usize = (0..6).filter(|&t| states[(0, t)] == 1).count();
    assert!(active_count >= 4, "Should have mostly active states");
}

#[test]
fn test_viterbi_binary_empty() {
    let prob = Array2::<f32>::zeros((0, 0));
    let trans = Array2::from_shape_vec((2, 2), vec![0.9, 0.1, 0.1, 0.9]).unwrap();

    let (states, logp) = viterbi_binary(&prob, &trans, None, None);
    assert_eq!(states.shape(), &[0, 0]);
    assert!(logp.is_empty());
}

#[test]
fn test_transition_uniform() {
    let trans = transition_uniform(3);
    assert_eq!(trans.shape(), &[3, 3]);

    // All transitions should be equal
    let expected = 1.0 / 3.0;
    for i in 0..3 {
        for j in 0..3 {
            assert!((trans[(i, j)] - expected).abs() < 0.01);
        }
    }
}

#[test]
fn test_transition_loop() {
    let trans = transition_loop(3, 0.7);
    assert_eq!(trans.shape(), &[3, 3]);

    // Diagonal should be 0.7
    for i in 0..3 {
        assert!((trans[(i, i)] - 0.7).abs() < 0.01);
    }

    // Off-diagonal should be (1-0.7)/2 = 0.15
    for i in 0..3 {
        for j in 0..3 {
            if i != j {
                assert!((trans[(i, j)] - 0.15).abs() < 0.01);
            }
        }
    }
}

#[test]
fn test_transition_local() {
    let trans = transition_local(4, 0.6);
    assert_eq!(trans.shape(), &[4, 4]);

    // Diagonal should be 0.6
    for i in 1..3 {
        assert!((trans[(i, i)] - 0.6).abs() < 0.01);
    }

    // Adjacent states should have (1-0.6)/2 = 0.2
    for i in 1..3 {
        assert!((trans[(i, i - 1)] - 0.2).abs() < 0.01);
        assert!((trans[(i, i + 1)] - 0.2).abs() < 0.01);
    }

    // Non-adjacent should be 0
    assert_eq!(trans[(0, 2)], 0.0);
    assert_eq!(trans[(1, 3)], 0.0);
}

#[test]
fn test_transition_cycle() {
    let trans = transition_cycle(4, 0.8);
    assert_eq!(trans.shape(), &[4, 4]);

    // Forward transitions (i -> i+1) should be 0.8
    for i in 0..3 {
        assert!((trans[(i, i + 1)] - 0.8).abs() < 0.01);
    }

    // Wrap around: 3 -> 0
    assert!((trans[(3, 0)] - 0.8).abs() < 0.01);
}

#[test]
fn test_transition_matrices_sum_to_one() {
    let trans_uniform = transition_uniform(5);
    let trans_loop = transition_loop(5, 0.7);
    let trans_local = transition_local(5, 0.5);
    let trans_cycle = transition_cycle(5, 0.6);

    for trans in &[trans_uniform, trans_loop, trans_local, trans_cycle] {
        for i in 0..5 {
            let row_sum: f32 = (0..5).map(|j| trans[(i, j)]).sum();
            assert!(
                (row_sum - 1.0).abs() < 0.01,
                "Transition matrix row should sum to 1.0, got {}",
                row_sum
            );
        }
    }
}

#[test]
fn test_nn_filter_shape() {
    let spec = Array2::from_shape_vec((4, 8), (0..32).map(|x| x as f32).collect()).unwrap();
    let filtered = nn_filter(&spec, None, "median", -1);

    assert_eq!(filtered.shape(), spec.shape());
}

#[test]
fn test_nn_filter_smoothing() {
    // Create signal and apply filtering
    let mut data = vec![1.0; 24];
    data[12] = 10.0; // Value in middle

    let spec = Array2::from_shape_vec((3, 8), data).unwrap();
    let filtered = nn_filter(&spec, None, "median", 1);

    assert_eq!(filtered.shape(), spec.shape());

    // Filter should produce reasonable output
    for val in filtered.iter() {
        assert!(val.is_finite(), "Filtered values should be finite");
    }
}

#[test]
fn test_nn_filter_with_recurrence() {
    let spec =
        Array2::from_shape_vec((2, 4), vec![1.0, 2.0, 1.0, 3.0, 1.0, 2.0, 1.0, 3.0]).unwrap();

    // Frames 0 and 2 are similar
    let mut rec = Array2::zeros((4, 4));
    for i in 0..4 {
        rec[(i, i)] = 1.0;
    }
    rec[(0, 2)] = 0.9;
    rec[(2, 0)] = 0.9;

    let filtered = nn_filter(&spec, Some(&rec), "mean", 1);

    assert_eq!(filtered.shape(), spec.shape());
}

#[test]
fn test_nn_filter_median_vs_mean() {
    let spec = Array2::from_shape_vec(
        (3, 5),
        vec![
            1.0, 2.0, 10.0, 2.0, 1.0, 1.0, 2.0, 10.0, 2.0, 1.0, 1.0, 2.0, 10.0, 2.0, 1.0,
        ],
    )
    .unwrap();

    let filtered_median = nn_filter(&spec, None, "median", 0);
    let filtered_mean = nn_filter(&spec, None, "mean", 0);

    assert_eq!(filtered_median.shape(), spec.shape());
    assert_eq!(filtered_mean.shape(), spec.shape());

    // Median should be more robust to outliers
    // Both should smooth the signal
}

#[test]
fn test_nn_filter_empty() {
    let spec = Array2::<f32>::zeros((0, 0));
    let filtered = nn_filter(&spec, None, "median", -1);

    assert_eq!(filtered.shape(), &[0, 0]);
}

#[test]
fn test_nn_filter_axis_time() {
    let spec = Array2::from_shape_vec((3, 4), (0..12).map(|x| x as f32).collect()).unwrap();
    let filtered = nn_filter(&spec, None, "median", 1);

    assert_eq!(filtered.shape(), spec.shape());
}

#[test]
fn test_nn_filter_axis_freq() {
    let spec = Array2::from_shape_vec((3, 4), (0..12).map(|x| x as f32).collect()).unwrap();
    let filtered = nn_filter(&spec, None, "median", 0);

    assert_eq!(filtered.shape(), spec.shape());
}

#[test]
fn test_timelag_filter_shape() {
    let spec = Array2::from_shape_vec((4, 8), (0..32).map(|x| x as f32).collect()).unwrap();
    let filtered = timelag_filter(&spec, None, "mean", None);

    assert_eq!(filtered.shape(), spec.shape());
}

#[test]
fn test_timelag_filter_repetitive() {
    // Repetitive pattern: [1,2,1,2,1,2,1,2]
    let spec = Array2::from_shape_vec(
        (2, 8),
        vec![
            1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0,
        ],
    )
    .unwrap();

    let filtered = timelag_filter(&spec, None, "mean", None);

    assert_eq!(filtered.shape(), spec.shape());

    // Filtered result should preserve repetitive structure
    for val in filtered.iter() {
        assert!(val.is_finite());
    }
}

#[test]
fn test_timelag_filter_with_lags() {
    let spec = Array2::from_shape_vec((3, 6), (0..18).map(|x| x as f32).collect()).unwrap();

    // Specify particular lags to keep
    let lags = vec![0, 1, 2];
    let filtered = timelag_filter(&spec, Some(&lags), "mean", None);

    assert_eq!(filtered.shape(), spec.shape());
}

#[test]
fn test_timelag_filter_aggregation() {
    let spec = Array2::from_shape_vec((2, 6), vec![1.0; 12]).unwrap();

    let filtered_mean = timelag_filter(&spec, None, "mean", None);
    let filtered_median = timelag_filter(&spec, None, "median", None);
    let filtered_max = timelag_filter(&spec, None, "max", None);

    assert_eq!(filtered_mean.shape(), spec.shape());
    assert_eq!(filtered_median.shape(), spec.shape());
    assert_eq!(filtered_max.shape(), spec.shape());
}

#[test]
fn test_timelag_filter_normalization() {
    let spec =
        Array2::from_shape_vec((2, 4), vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]).unwrap();

    let filtered_l1 = timelag_filter(&spec, None, "mean", Some("L1"));
    let filtered_l2 = timelag_filter(&spec, None, "mean", Some("L2"));

    assert_eq!(filtered_l1.shape(), spec.shape());
    assert_eq!(filtered_l2.shape(), spec.shape());

    // Check L1 normalization (each row sums to 1)
    for f in 0..2 {
        let sum: f32 = (0..4).map(|t| filtered_l1[(f, t)].abs()).sum();
        if sum > 0.0 {
            assert!(
                (sum - 1.0).abs() < 0.1,
                "L1 norm should be ~1.0, got {}",
                sum
            );
        }
    }
}

#[test]
fn test_timelag_filter_empty() {
    let spec = Array2::<f32>::zeros((0, 0));
    let filtered = timelag_filter(&spec, None, "mean", None);

    assert_eq!(filtered.shape(), &[0, 0]);
}

#[test]
fn test_binary_mask() {
    let ref_spec = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 0.5, 1.0, 3.0, 0.8]).unwrap();
    let other = Array2::from_shape_vec((2, 3), vec![0.5, 1.5, 1.0, 0.8, 2.0, 1.5]).unwrap();

    let mask = binary_mask(&ref_spec, &[&other]);

    assert_eq!(mask.shape(), ref_spec.shape());

    // Check that mask is binary
    for val in mask.iter() {
        assert!(*val == 0.0 || *val == 1.0);
    }

    // First bin: ref (1.0) > other (0.5), so mask should be 1
    assert_eq!(mask[(0, 0)], 1.0);

    // Second bin: ref (2.0) > other (1.5), so mask should be 1
    assert_eq!(mask[(0, 1)], 1.0);

    // Third bin: ref (0.5) < other (1.0), so mask should be 0
    assert_eq!(mask[(0, 2)], 0.0);
}

#[test]
fn test_split_softmask() {
    let s1 = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]).unwrap();
    let s2 = Array2::from_shape_vec((2, 3), vec![0.5, 1.0, 1.5, 0.5, 1.0, 1.5]).unwrap();

    let masks = split_softmask(&[&s1, &s2], 1.0, true);

    assert_eq!(masks.len(), 2);
    assert_eq!(masks[0].shape(), s1.shape());
    assert_eq!(masks[1].shape(), s2.shape());

    // Masks should sum to 1 at each bin
    for i in 0..2 {
        for j in 0..3 {
            let sum = masks[0][(i, j)] + masks[1][(i, j)];
            assert!(
                (sum - 1.0).abs() < 0.01,
                "Masks should sum to 1, got {}",
                sum
            );
        }
    }

    // s1 is 2x larger than s2, so mask[0] should be ~0.67, mask[1] ~0.33
    assert!((masks[0][(0, 0)] - 0.67).abs() < 0.1);
    assert!((masks[1][(0, 0)] - 0.33).abs() < 0.1);
}

#[test]
fn test_split_softmask_zeros() {
    let s1 = Array2::from_shape_vec((2, 2), vec![0.0, 1.0, 0.0, 1.0]).unwrap();
    let s2 = Array2::from_shape_vec((2, 2), vec![0.0, 2.0, 0.0, 2.0]).unwrap();

    let masks = split_softmask(&[&s1, &s2], 1.0, true);

    // Where both are zero, should split evenly
    assert!((masks[0][(0, 0)] - 0.5).abs() < 0.01);
    assert!((masks[1][(0, 0)] - 0.5).abs() < 0.01);

    // Where both are non-zero, should follow magnitude
    assert!(masks[1][(0, 1)] > masks[0][(0, 1)]);
}

#[test]
fn test_power_softmask() {
    let reference = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 0.5, 1.0, 2.0, 0.5]).unwrap();
    let mixture = Array2::from_shape_vec((2, 3), vec![2.0, 3.0, 1.0, 2.0, 3.0, 1.0]).unwrap();

    let mask = power_softmask(&reference, &mixture, 1.0);

    assert_eq!(mask.shape(), reference.shape());

    // All mask values should be in [0, 1]
    for val in mask.iter() {
        assert!(*val >= 0.0 && *val <= 1.0);
    }

    // First bin: ref/mix = 1/2 = 0.5
    assert!((mask[(0, 0)] - 0.5).abs() < 0.01);

    // Second bin: ref/mix = 2/3 ~= 0.67
    assert!((mask[(0, 1)] - 0.67).abs() < 0.1);
}

#[test]
fn test_power_softmask_power_parameter() {
    let reference = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap();
    let mixture = Array2::from_shape_vec((1, 2), vec![2.0, 3.0]).unwrap();

    let mask_p1 = power_softmask(&reference, &mixture, 1.0);
    let mask_p2 = power_softmask(&reference, &mixture, 2.0);

    // Higher power should give different results
    assert!(mask_p1[(0, 0)] != mask_p2[(0, 0)]);
}

#[test]
fn test_split_softmask_empty() {
    let masks = split_softmask(&[], 1.0, true);
    assert_eq!(masks.len(), 0);
}

#[test]
fn test_binary_mask_multiple_sources() {
    let ref_spec = Array2::from_shape_vec((2, 2), vec![2.0, 1.0, 2.0, 1.0]).unwrap();
    let other1 = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 1.0, 2.0]).unwrap();
    let other2 = Array2::from_shape_vec((2, 2), vec![0.5, 0.5, 0.5, 0.5]).unwrap();

    let mask = binary_mask(&ref_spec, &[&other1, &other2]);

    // First bin: ref (2.0) > other1 (1.0) > other2 (0.5), mask = 1
    assert_eq!(mask[(0, 0)], 1.0);

    // Second bin: other1 (2.0) > ref (1.0) > other2 (0.5), mask = 0
    assert_eq!(mask[(0, 1)], 0.0);
}

#[test]
fn test_rqa_identity_matrix() {
    // Identity matrix: only diagonal recurrence
    let mut rec = Array2::zeros((5, 5));
    for i in 0..5 {
        rec[(i, i)] = 1.0;
    }

    let (det, lam, ent) = rqa(&rec);

    // All metrics should be in valid range
    assert!((0.0..=1.0).contains(&det));
    assert!((0.0..=1.0).contains(&lam));
    assert!(ent >= 0.0);
}

#[test]
fn test_rqa_full_recurrence() {
    // Full recurrence matrix
    let rec = Array2::from_elem((6, 6), 1.0);

    let (det, lam, ent) = rqa(&rec);

    // Full recurrence should have high determinism and laminarity
    assert!(det > 0.5, "Full recurrence should have high determinism");
    assert!(lam > 0.5, "Full recurrence should have high laminarity");
    assert!(ent >= 0.0);
}

#[test]
fn test_rqa_diagonal_structure() {
    // Diagonal structure (repetitive pattern)
    let mut rec = Array2::zeros((8, 8));
    for i in 0..8 {
        for j in 0..8 {
            // Create diagonal lines
            if (i as i32 - j as i32).abs() <= 1 {
                rec[(i, j)] = 1.0;
            }
        }
    }

    let (det, _lam, _ent) = rqa(&rec);

    // Should have high determinism due to diagonal structures
    assert!(
        det > 0.3,
        "Diagonal structure should have high determinism, got {}",
        det
    );
}

#[test]
fn test_rqa_empty() {
    let rec = Array2::<f32>::zeros((0, 0));
    let (det, lam, ent) = rqa(&rec);

    assert_eq!(det, 0.0);
    assert_eq!(lam, 0.0);
    assert_eq!(ent, 0.0);
}

#[test]
fn test_rqa_no_recurrence() {
    // No recurrence at all
    let rec = Array2::zeros((5, 5));

    let (det, lam, ent) = rqa(&rec);

    assert_eq!(det, 0.0);
    assert_eq!(lam, 0.0);
    assert_eq!(ent, 0.0);
}

#[test]
fn test_rqa_detailed() {
    // Create a simple recurrence pattern
    let mut rec = Array2::zeros((6, 6));
    for i in 0..6 {
        rec[(i, i)] = 1.0;
    }
    // Add some diagonal structure
    for i in 0..4 {
        rec[(i, i + 2)] = 1.0;
        rec[(i + 2, i)] = 1.0;
    }

    let (det, lam, ent, avg_len) = rqa_detailed(&rec, 2);

    assert!((0.0..=1.0).contains(&det));
    assert!((0.0..=1.0).contains(&lam));
    assert!(ent >= 0.0);
    assert!(
        avg_len >= 0.0,
        "Average diagonal length should be non-negative"
    );
}

#[test]
fn test_rqa_detailed_min_length() {
    let mut rec = Array2::zeros((8, 8));
    // Create isolated points (no diagonal lines >= 2)
    for i in 0..7 {
        rec[(i, i + 1)] = 1.0;
    }

    // With min_length = 1, should find these
    let (det1, _, _, _) = rqa_detailed(&rec, 1);
    // With min_length = 5, should find very few or none
    let (det5, _, _, _) = rqa_detailed(&rec, 5);

    assert!(det1 > 0.0, "Should detect short lines with min_length=1");
    // With higher min_length, determinism should be lower or zero
    assert!(
        det5 <= det1,
        "Higher min_length should give lower or equal determinism"
    );
}

#[test]
fn test_rqa_metrics_bounds() {
    // Random binary matrix
    let mut rec = Array2::zeros((10, 10));
    for i in 0..10 {
        for j in 0..10 {
            if (i + j) % 3 == 0 {
                rec[(i, j)] = 1.0;
            }
        }
    }

    let (det, lam, ent) = rqa(&rec);

    // All metrics should be non-negative and determinism/laminarity <= 1
    assert!(
        (0.0..=1.0).contains(&det),
        "Determinism out of bounds: {}",
        det
    );
    assert!(
        (0.0..=1.0).contains(&lam),
        "Laminarity out of bounds: {}",
        lam
    );
    assert!(ent >= 0.0, "Entropy should be non-negative: {}", ent);
}

#[test]
fn test_nmf_basic() {
    let v = Array2::from_shape_vec(
        (4, 6),
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.0, 1.0, 2.0, 2.0, 3.0,
            3.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
        ],
    )
    .unwrap();

    let (w, h) = nmf(&v, 2, Some(100), None);

    assert_eq!(w.shape(), &[4, 2]);
    assert_eq!(h.shape(), &[2, 6]);

    // All values should be non-negative
    for &val in w.iter() {
        assert!(val >= 0.0, "W should be non-negative");
    }
    for &val in h.iter() {
        assert!(val >= 0.0, "H should be non-negative");
    }
}

#[test]
fn test_nmf_reconstruction() {
    let v = Array2::from_shape_vec(
        (3, 4),
        vec![1.0, 2.0, 1.0, 2.0, 2.0, 4.0, 2.0, 4.0, 3.0, 6.0, 3.0, 6.0],
    )
    .unwrap();

    let (w, h) = nmf(&v, 2, Some(200), None);

    // Reconstruct
    let mut v_approx = Array2::<f32>::zeros((3, 4));
    for i in 0..3 {
        for j in 0..4 {
            for k in 0..2 {
                v_approx[(i, j)] += w[(i, k)] * h[(k, j)];
            }
        }
    }

    // Check reconstruction error is reasonable
    let mut error = 0.0f32;
    for i in 0..3 {
        for j in 0..4 {
            error += (v[(i, j)] - v_approx[(i, j)]).abs();
        }
    }
    let avg_error = error / 12.0;
    assert!(
        avg_error < 2.0,
        "Reconstruction error {} too high",
        avg_error
    );
}

#[test]
fn test_nmf_empty() {
    let v = Array2::<f32>::zeros((0, 0));
    let (w, h) = nmf(&v, 2, None, None);
    assert_eq!(w.shape(), &[0, 0]);
    assert_eq!(h.shape(), &[0, 0]);
}

#[test]
fn test_decompose_basic() {
    let spec = Array2::from_shape_vec(
        (4, 8),
        vec![
            1.0, 2.0, 1.5, 2.5, 1.0, 2.0, 1.5, 2.5, 2.0, 3.0, 2.5, 3.5, 2.0, 3.0, 2.5, 3.5, 0.5,
            1.0, 0.8, 1.2, 0.5, 1.0, 0.8, 1.2, 1.5, 2.5, 2.0, 3.0, 1.5, 2.5, 2.0, 3.0,
        ],
    )
    .unwrap();

    let (comps, acts) = decompose(&spec, Some(2), None, false);

    assert_eq!(comps.shape(), &[4, 2]);
    assert_eq!(acts.shape(), &[2, 8]);
}

#[test]
fn test_decompose_sorted() {
    let spec = Array2::from_shape_vec(
        (4, 8),
        vec![
            0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 1.0, 2.0, 1.5, 2.5, 1.0, 2.0, 1.5, 2.5, 0.5,
            1.0, 0.8, 1.2, 0.5, 1.0, 0.8, 1.2, 2.0, 3.0, 2.5, 3.5, 2.0, 3.0, 2.5, 3.5,
        ],
    )
    .unwrap();

    let (comps, _acts) = decompose(&spec, Some(2), None, true);

    // Components should be sorted by peak frequency
    let mut prev_peak = 0;
    for k in 0..2 {
        let mut max_val = 0.0f32;
        let mut peak_idx = 0;
        for i in 0..4 {
            if comps[(i, k)] > max_val {
                max_val = comps[(i, k)];
                peak_idx = i;
            }
        }
        if k > 0 {
            assert!(
                peak_idx >= prev_peak,
                "Components should be sorted by peak frequency"
            );
        }
        prev_peak = peak_idx;
    }
}

#[test]
fn test_decompose_empty() {
    let spec = Array2::<f32>::zeros((0, 0));
    let (comps, acts) = decompose(&spec, Some(2), None, false);
    assert_eq!(comps.shape(), &[0, 0]);
    assert_eq!(acts.shape(), &[0, 0]);
}

#[test]
fn test_nmf_hpss_basic() {
    let spec = Array2::from_shape_vec(
        (8, 10),
        (0..80)
            .map(|i| (i as f32 * 0.1).sin().abs() + 0.1)
            .collect(),
    )
    .unwrap();

    let (harmonic, percussive) = nmf_hpss(&spec, None, None, None);

    assert_eq!(harmonic.shape(), spec.shape());
    assert_eq!(percussive.shape(), spec.shape());

    // All values should be non-negative
    for &val in harmonic.iter() {
        assert!(val >= 0.0);
    }
    for &val in percussive.iter() {
        assert!(val >= 0.0);
    }
}

#[test]
fn test_nmf_hpss_sum() {
    let spec = Array2::from_shape_vec(
        (4, 6),
        (0..24).map(|i| (i as f32 * 0.2).abs() + 0.5).collect(),
    )
    .unwrap();

    let (harmonic, percussive) = nmf_hpss(&spec, Some(2), Some(2), Some(50));

    // Harmonic + Percussive should approximate original
    // (not exact due to NMF approximation)
    let mut total_diff = 0.0f32;
    let mut total_sum = 0.0f32;
    for i in 0..4 {
        for j in 0..6 {
            let sum = harmonic[(i, j)] + percussive[(i, j)];
            total_diff += (spec[(i, j)] - sum).abs();
            total_sum += spec[(i, j)];
        }
    }

    // Relative error should be reasonable
    let rel_error = total_diff / total_sum.max(1.0);
    assert!(
        rel_error < 1.0,
        "Relative reconstruction error {} too high",
        rel_error
    );
}

#[test]
fn test_nmf_hpss_empty() {
    let spec = Array2::<f32>::zeros((0, 0));
    let (harmonic, percussive) = nmf_hpss(&spec, None, None, None);
    assert_eq!(harmonic.shape(), &[0, 0]);
    assert_eq!(percussive.shape(), &[0, 0]);
}

#[test]
fn test_agglomerative_basic() {
    // Create features with distinct segments
    let mut data = Array2::<f32>::zeros((4, 12));
    // First segment: frames 0-3 with similar values
    for t in 0..4 {
        for f in 0..4 {
            data[(f, t)] = 1.0 + (f as f32) * 0.1;
        }
    }
    // Second segment: frames 4-7 with different values
    for t in 4..8 {
        for f in 0..4 {
            data[(f, t)] = 5.0 + (f as f32) * 0.1;
        }
    }
    // Third segment: frames 8-11 with yet different values
    for t in 8..12 {
        for f in 0..4 {
            data[(f, t)] = 10.0 + (f as f32) * 0.1;
        }
    }

    let bounds = agglomerative(&data, 3);
    assert_eq!(bounds.len(), 3);
    assert_eq!(bounds[0], 0);
    // Should roughly identify the segment transitions
}

#[test]
fn test_agglomerative_k_equals_frames() {
    let data = Array2::from_shape_vec(
        (2, 5),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0],
    )
    .unwrap();

    let bounds = agglomerative(&data, 5);
    assert_eq!(bounds, vec![0, 1, 2, 3, 4]);
}

#[test]
fn test_agglomerative_k_greater_than_frames() {
    let data = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]).unwrap();

    let bounds = agglomerative(&data, 10);
    assert_eq!(bounds, vec![0, 1, 2]);
}

#[test]
fn test_agglomerative_k_one() {
    let data =
        Array2::from_shape_vec((2, 4), vec![1.0, 5.0, 2.0, 6.0, 1.0, 5.0, 2.0, 6.0]).unwrap();

    let bounds = agglomerative(&data, 1);
    assert_eq!(bounds, vec![0]);
}

#[test]
fn test_agglomerative_empty() {
    let data = Array2::<f32>::zeros((0, 0));
    let bounds = agglomerative(&data, 3);
    assert_eq!(bounds, vec![0]);
}

#[test]
fn test_subsegment_basic() {
    let features =
        Array2::from_shape_vec((4, 20), (0..80).map(|x| (x as f32 * 0.5).sin()).collect()).unwrap();

    let frames = vec![0, 10, 20];
    let sub_bounds = subsegment(&features, &frames, 2);

    // Should have more boundaries than original
    assert!(sub_bounds.len() >= 2);
    assert_eq!(sub_bounds[0], 0);
    // All boundaries should be valid
    for &b in &sub_bounds {
        assert!(b <= 20);
    }
}

#[test]
fn test_subsegment_n_segments_one() {
    let features = Array2::from_shape_vec((2, 10), (0..20).map(|x| x as f32).collect()).unwrap();

    let frames = vec![0, 5, 10];
    let sub_bounds = subsegment(&features, &frames, 1);

    // With n_segments=1, should just return the input frames
    assert_eq!(sub_bounds[0], 0);
    assert!(sub_bounds.contains(&5));
}

#[test]
fn test_subsegment_empty_frames() {
    let features = Array2::from_shape_vec((2, 10), (0..20).map(|x| x as f32).collect()).unwrap();

    let sub_bounds = subsegment(&features, &[], 2);
    assert_eq!(sub_bounds, vec![0]);
}

#[test]
fn test_subsegment_empty_data() {
    let features = Array2::<f32>::zeros((0, 0));
    let frames = vec![0, 5, 10];
    let sub_bounds = subsegment(&features, &frames, 2);
    assert_eq!(sub_bounds, vec![0]);
}

#[test]
fn test_path_enhance_basic() {
    // Create a simple recurrence-like matrix with a diagonal
    let mut r = Array2::<f32>::zeros((20, 20));
    for i in 0..20 {
        r[(i, i)] = 1.0;
        if i > 0 {
            r[(i, i - 1)] = 0.5;
        }
        if i < 19 {
            r[(i, i + 1)] = 0.5;
        }
    }

    let enhanced = path_enhance(&r, 5, 2.0, None, 3, false, true);

    assert_eq!(enhanced.shape(), r.shape());
    // Enhanced diagonal should still be strong
    for i in 0..20 {
        assert!(enhanced[(i, i)] > 0.0);
    }
}

#[test]
fn test_path_enhance_with_min_ratio() {
    let r = Array2::from_shape_vec(
        (10, 10),
        (0..100)
            .map(|i| if i / 10 == i % 10 { 1.0 } else { 0.0 })
            .collect(),
    )
    .unwrap();

    let enhanced = path_enhance(&r, 3, 2.0, Some(0.5), 5, false, true);
    assert_eq!(enhanced.shape(), &[10, 10]);
}

#[test]
fn test_path_enhance_zero_mean() {
    let r = Array2::from_shape_vec((8, 8), (0..64).map(|_| 0.5).collect()).unwrap();

    let enhanced = path_enhance(&r, 5, 2.0, None, 3, true, false);
    assert_eq!(enhanced.shape(), &[8, 8]);
}

#[test]
fn test_path_enhance_clip() {
    let r = Array2::from_shape_vec((6, 6), (0..36).map(|i| (i as f32 - 18.0) / 36.0).collect())
        .unwrap();

    let enhanced = path_enhance(&r, 3, 2.0, None, 2, true, true);

    // All values should be non-negative after clipping
    for &val in enhanced.iter() {
        assert!(val >= 0.0);
    }
}

#[test]
fn test_path_enhance_empty() {
    let r = Array2::<f32>::zeros((0, 0));
    let enhanced = path_enhance(&r, 5, 2.0, None, 3, false, true);
    assert_eq!(enhanced.shape(), &[0, 0]);
}

#[test]
fn test_path_enhance_single_filter() {
    let r = Array2::from_shape_vec(
        (5, 5),
        (0..25)
            .map(|i| if i / 5 == i % 5 { 1.0 } else { 0.1 })
            .collect(),
    )
    .unwrap();

    let enhanced = path_enhance(&r, 3, 2.0, None, 1, false, true);
    assert_eq!(enhanced.shape(), &[5, 5]);
}

#[test]
fn test_frame_basic() {
    // Test basic framing: data = [0, 1, 2, 3, 4, 5, 6], frame_length=3, hop_length=2
    // Expected frames (as columns):
    // Column 0: [0, 1, 2]
    // Column 1: [2, 3, 4]
    // Column 2: [4, 5, 6]
    let data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let frames = frame(&data, 3, 2).unwrap();

    assert_eq!(frames.shape(), &[3, 3]);

    // Check first frame
    assert_eq!(frames[(0, 0)], 0.0);
    assert_eq!(frames[(1, 0)], 1.0);
    assert_eq!(frames[(2, 0)], 2.0);

    // Check second frame
    assert_eq!(frames[(0, 1)], 2.0);
    assert_eq!(frames[(1, 1)], 3.0);
    assert_eq!(frames[(2, 1)], 4.0);

    // Check third frame
    assert_eq!(frames[(0, 2)], 4.0);
    assert_eq!(frames[(1, 2)], 5.0);
    assert_eq!(frames[(2, 2)], 6.0);
}

#[test]
fn test_frame_hop_1() {
    // Test with hop_length=1 (maximum overlap)
    let data = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let frames = frame(&data, 3, 1).unwrap();

    // n_frames = 1 + (5 - 3) / 1 = 3
    assert_eq!(frames.shape(), &[3, 3]);

    // Check all frames
    assert_eq!(frames.column(0).to_vec(), vec![0.0, 1.0, 2.0]);
    assert_eq!(frames.column(1).to_vec(), vec![1.0, 2.0, 3.0]);
    assert_eq!(frames.column(2).to_vec(), vec![2.0, 3.0, 4.0]);
}

#[test]
fn test_frame_exact_fit() {
    // Test when data length exactly fits the frames
    let data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let frames = frame(&data, 3, 3).unwrap();

    // n_frames = 1 + (6 - 3) / 3 = 2
    assert_eq!(frames.shape(), &[3, 2]);

    assert_eq!(frames.column(0).to_vec(), vec![0.0, 1.0, 2.0]);
    assert_eq!(frames.column(1).to_vec(), vec![3.0, 4.0, 5.0]);
}

#[test]
fn test_frame_error_too_short() {
    let data = vec![0.0, 1.0]; // Too short for frame_length=3
    let result = frame(&data, 3, 1);
    assert!(result.is_err());
}

#[test]
fn test_frame_error_zero_frame_length() {
    let data = vec![0.0, 1.0, 2.0];
    let result = frame(&data, 0, 1);
    assert!(result.is_err());
}

#[test]
fn test_frame_error_zero_hop_length() {
    let data = vec![0.0, 1.0, 2.0];
    let result = frame(&data, 2, 0);
    assert!(result.is_err());
}

#[test]
fn test_frame_single_frame() {
    // When data length == frame_length, should return single frame
    let data = vec![0.0, 1.0, 2.0];
    let frames = frame(&data, 3, 1).unwrap();

    assert_eq!(frames.shape(), &[3, 1]);
    assert_eq!(frames.column(0).to_vec(), vec![0.0, 1.0, 2.0]);
}
