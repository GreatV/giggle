use ndarray::Array2;

/// Fill off-diagonal elements of a 2D array with a specified value.
/// Diagonal elements (where i == j) are left unchanged.
///
/// # Arguments
/// * `arr` - Input array to modify (in-place operation returns new array)
/// * `value` - Value to fill off-diagonal elements with
pub fn fill_off_diagonal(arr: &Array2<f32>, value: f32) -> Array2<f32> {
    let mut result = arr.clone();
    let (rows, cols) = (arr.shape()[0], arr.shape()[1]);

    for i in 0..rows {
        for j in 0..cols {
            if i != j {
                result[(i, j)] = value;
            }
        }
    }

    result
}

/// Shear a 2D array along the specified axis.
///
/// This shifts each row (or column) by an amount proportional to its index.
///
/// # Arguments
/// * `arr` - Input 2D array
/// * `factor` - Shearing factor
/// * `axis` - Axis along which to shear (0 = shear rows, 1 = shear columns)
///
/// # Returns
/// Sheared array with same shape
pub fn shear(arr: &Array2<f32>, factor: f32, axis: usize) -> Array2<f32> {
    let shape = arr.shape();
    let (rows, cols) = (shape[0], shape[1]);
    let mut result = Array2::<f32>::zeros((rows, cols));

    if axis == 0 {
        // Shear rows: shift each row based on its row index
        for i in 0..rows {
            let shift = (i as f32 * factor).round() as isize;
            for j in 0..cols {
                let src_j = (j as isize - shift).rem_euclid(cols as isize) as usize;
                result[(i, j)] = arr[(i, src_j)];
            }
        }
    } else {
        // Shear columns: shift each column based on its column index
        for j in 0..cols {
            let shift = (j as f32 * factor).round() as isize;
            for i in 0..rows {
                let src_i = (i as isize - shift).rem_euclid(rows as isize) as usize;
                result[(i, j)] = arr[(src_i, j)];
            }
        }
    }

    result
}

/// Compute gradient with cyclic (circular) boundary conditions.
///
/// Computes the gradient using centered differences with wraparound at boundaries.
///
/// # Arguments
/// * `arr` - Input array
/// * `axis` - Axis along which to compute gradient (0 for rows, 1 for columns)
///
/// # Returns
/// Gradient array with same shape as input
pub fn cyclic_gradient(arr: &Array2<f32>, axis: usize) -> Array2<f32> {
    let shape = arr.shape();
    let (rows, cols) = (shape[0], shape[1]);
    let mut result = Array2::<f32>::zeros((rows, cols));

    if axis == 0 {
        // Gradient along rows (vertical)
        for i in 0..rows {
            let prev = (i + rows - 1) % rows;
            let next = (i + 1) % rows;
            for j in 0..cols {
                result[(i, j)] = (arr[(next, j)] - arr[(prev, j)]) / 2.0;
            }
        }
    } else {
        // Gradient along columns (horizontal)
        for j in 0..cols {
            let prev = (j + cols - 1) % cols;
            let next = (j + 1) % cols;
            for i in 0..rows {
                result[(i, j)] = (arr[(i, next)] - arr[(i, prev)]) / 2.0;
            }
        }
    }

    result
}

/// Stack arrays along a new axis.
///
/// # Arguments
/// * `arrays` - Slice of 2D arrays to stack
/// * `axis` - New axis position (0 = stack as first dimension)
///
/// # Returns
/// Stacked array with one additional dimension
///
/// Note: This is a simplified version that stacks 2D arrays
pub fn stack(arrays: &[&Array2<f32>], axis: usize) -> crate::Result<Array2<f32>> {
    if arrays.is_empty() {
        return Ok(Array2::<f32>::zeros((0, 0)));
    }

    let first_shape = arrays[0].shape();
    let (rows, cols) = (first_shape[0], first_shape[1]);

    // Verify all arrays have the same shape
    for arr in arrays {
        if arr.shape() != first_shape {
            return Err(crate::Error::ShapeMismatch {
                expected: format!("{:?}", first_shape),
                got: format!("{:?}", arr.shape()),
            });
        }
    }

    if axis == 0 {
        // Stack along rows: concatenate vertically
        let total_rows = rows * arrays.len();
        let mut result = Array2::<f32>::zeros((total_rows, cols));

        for (idx, arr) in arrays.iter().enumerate() {
            let start_row = idx * rows;
            for i in 0..rows {
                for j in 0..cols {
                    result[(start_row + i, j)] = arr[(i, j)];
                }
            }
        }
        Ok(result)
    } else {
        // Stack along columns: concatenate horizontally
        let total_cols = cols * arrays.len();
        let mut result = Array2::<f32>::zeros((rows, total_cols));

        for (idx, arr) in arrays.iter().enumerate() {
            let start_col = idx * cols;
            for i in 0..rows {
                for j in 0..cols {
                    result[(i, start_col + j)] = arr[(i, j)];
                }
            }
        }
        Ok(result)
    }
}

/// Count the number of unique values in an array.
///
/// # Arguments
/// * `arr` - Input array
///
/// # Returns
/// Number of unique values
pub fn count_unique(arr: &[f32]) -> usize {
    use std::collections::HashSet;

    let mut seen = HashSet::new();
    for &val in arr {
        // Use ordered bit representation for float comparison
        seen.insert(val.to_bits());
    }
    seen.len()
}

/// Check if all values in an array are unique.
///
/// # Arguments
/// * `arr` - Input array
///
/// # Returns
/// true if all values are unique, false otherwise
pub fn is_unique(arr: &[f32]) -> bool {
    count_unique(arr) == arr.len()
}
