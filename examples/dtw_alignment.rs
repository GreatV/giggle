//! Dynamic Time Warping (DTW) Example
//!
//! This example demonstrates sequence alignment using DTW.

use giggle::utils::dtw;
use log::info;
use ndarray::Array2;

fn main() {
    env_logger::init();
    info!("Dynamic Time Warping (DTW) Example");

    // Create two similar but time-warped sequences
    let x: Vec<f32> = (0..50).map(|i| (i as f32 * 0.2).sin()).collect();
    let y: Vec<f32> = (0..40).map(|i| (i as f32 * 0.25).sin()).collect();

    info!("Sequence X: {} samples", x.len());
    info!("Sequence Y: {} samples", y.len());

    // Convert to 2D arrays
    let x_arr = Array2::from_shape_vec((1, x.len()), x.clone()).unwrap();
    let y_arr = Array2::from_shape_vec((1, y.len()), y.clone()).unwrap();

    // Basic DTW
    info!("Basic DTW (Euclidean)");

    let (distance, path) = dtw(&x_arr, &y_arr, "euclidean");

    info!("DTW distance: {:.4}", distance);
    info!("Warp path length: {} points", path.len());

    // Show first few path points
    info!("\nFirst 10 path points (X -> Y):");
    for (i, &(xi, yi)) in path.iter().take(10).enumerate() {
        info!("  Step {}: X[{}] -> Y[{}]", i, xi, yi);
    }

    // Different Distance Metrics
    info!("Different Distance Metrics");

    for metric in ["euclidean", "manhattan", "cosine"] {
        let (dist, _) = dtw(&x_arr, &y_arr, metric);
        info!("  {}: distance={:.4}", metric, dist);
    }
}
