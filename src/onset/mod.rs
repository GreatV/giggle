pub mod detect;
pub mod strength;

pub use detect::{OnsetDetectConfig, onset_backtrack, onset_detect};
pub use strength::onset_strength;
