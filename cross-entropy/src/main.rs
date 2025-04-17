use ndarray::prelude::*;
use ndarray::Array1;
use std::f32;


// Write a function that takes as input two lists Y, P,
// and returns the float corresponding to their cross-entropy.

// Define a small epsilon for numerical stability to avoid log(0)
const EPSILON: f32 = 1e-15;


// Binary Cross-Entropy Loss Function
// Formula: CE = - Σ [ yᵢ * ln(pᵢ) + (1 - yᵢ) * ln(1 - pᵢ) ]
// Takes references to avoid consuming the input arrays.
fn cross_entropy(y: &Array1<f32>, p: &Array1<f32>) -> f32 {

    // Ensure inputs have the same length
    assert_eq!(y.len(), p.len(), "Input arrays Y and P must have the same length.");

    // Calculate the first term: y * ln(p + ε)
    // Add EPSILON inside the ln to avoid ln(0) -> -inf
    let term1 = y * &(p + EPSILON).mapv(f32::ln);

    // Calculate the second term: (1 - y) * ln(1 - p + ε)
    // Add EPSILON inside the ln to avoid ln(0) -> -inf
    let term2 = (1.0 - y) * &(1.0 - p + EPSILON).mapv(f32::ln); // mapv applies ln element-wise


    // Sum the terms element-wise, then sum all elements in the resulting array,
    // and finally negate the result according to the formula.
    - (term1 + term2).sum()
}

// --- Alternative using .dot (can be less numerically stable if not careful) ---
// This version uses dot product directly but requires careful handling of intermediates.
// The element-wise version above is often clearer.
fn cross_entropy_dot(y: &Array1<f32>, p: &Array1<f32>) -> f32 {
    assert_eq!(y.len(), p.len(), "Input arrays Y and P must have the same length.");

    // Calculate ln(p + ε)
    let log_p = (p + EPSILON).mapv(f32::ln);

    // Calculate ln(1 - p + ε)
    let log_1_minus_p = (1.0 - p + EPSILON).mapv(f32::ln);

    // Calculate Σ yᵢ * ln(pᵢ + ε)
    let present = y.dot(&log_p);

    // Calculate Σ (1 - yᵢ) * ln(1 - pᵢ + ε)
    // Need to compute (1.0 - y) first, which creates a temporary array
    let no_present = (1.0 - y).dot(&log_1_minus_p);

    // Apply the negative sign to the sum
    -(present + no_present)
}


fn main() {
    // Example usage
    let y = array![1.0, 0.0, 1.0]; // True labels
    let p = array![0.9, 0.1, 0.8]; // Predicted probabilities

    // Use the corrected function (taking references)
    let ce = cross_entropy(&y, &p);
    println!("Cross-entropy (element-wise method): {}", ce);

    let ce_dot = cross_entropy_dot(&y, &p);
    println!("Cross-entropy (dot product method): {}", ce_dot);

    // Expected calculation breakdown:
    // Term 1: 1.0 * ln(0.9) + 0.0 * ln(0.1) + 1.0 * ln(0.8)
    // Term 2: (1.0-1.0)*ln(1-0.9) + (1.0-0.0)*ln(1-0.1) + (1.0-1.0)*ln(1-0.8)
    // Term 1 ≈ 1.0 * (-0.10536) + 0.0 * (-2.30258) + 1.0 * (-0.22314) = -0.10536 - 0.22314 = -0.3285
    // Term 2 ≈ 0.0 * ln(0.1) + 1.0 * ln(0.9) + 0.0 * ln(0.2) = 0.0 + 1.0 * (-0.10536) + 0.0 = -0.10536
    // Sum = Term1 + Term2 ≈ -0.3285 - 0.10536 = -0.43386
    // Final CE = - Sum ≈ 0.43386
    // (The exact values from the code will be slightly different due to f32 precision and EPSILON)
}
