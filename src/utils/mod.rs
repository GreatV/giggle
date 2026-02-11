mod array;
mod complex;
mod dtw;
mod filtering;
mod framing;
mod hmm;
mod masking;
mod matching;
mod matrix;
mod numerical;
mod rqa;
mod segmentation;
mod similarity;
mod validation;

// Array operations: normalize, localmax, localmin, axis_sort, sparsify_rows, peak_pick
pub use array::{
    NormType, SparseRows, axis_sort, localmax, localmin, localmin_2d, normalize, normalize_2d,
    peak_pick, sparsify_rows, sync,
};

// Complex number operations
pub use complex::{abs2, phasor};

// Dynamic Time Warping
pub use dtw::{dtw, dtw_backtracking, dtw_distance, dtw_with_steps};

// Filtering operations
pub use filtering::{nn_filter, timelag_filter};

// Framing operations
pub use framing::{expand_to, fix_length, frame, frame_array, frame_count, pad_center, tiny};

// HMM (Hidden Markov Model) operations
pub use hmm::{
    transition_cycle, transition_local, transition_loop, transition_uniform, viterbi,
    viterbi_binary, viterbi_discriminative,
};

// Masking operations
pub use masking::{binary_mask, power_softmask, softmask, split_softmask};

// Matching operations
pub use matching::{match_events, match_intervals};

// Matrix operations
pub use matrix::{count_unique, cyclic_gradient, fill_off_diagonal, is_unique, shear, stack};

// Numerical operations
pub use numerical::{decompose, nmf, nmf_hpss, nnls};

// Recurrence quantification analysis
pub use rqa::{rqa, rqa_detailed};

// Segmentation operations
pub use segmentation::{agglomerative, path_enhance, subsegment};

// Similarity operations
pub use similarity::{cross_similarity, lag_to_recurrence, recurrence_matrix, recurrence_to_lag};

// Validation operations
pub use validation::{
    autocorrelate, buf_to_float, fix_frames, fix_frames_f32, i16_to_float, i32_to_float,
    index_to_slice, index_to_slice_f32, is_positive_int, mse, valid_audio, valid_audio_2d,
    valid_int, valid_intervals, zero_crossings,
};

#[cfg(test)]
mod tests;
