```markdown
# Binary Cross-Entropy Calculation in Rust

This Rust code provides a function to calculate the binary cross-entropy loss between true labels and predicted probabilities. This loss function is commonly used in machine learning, particularly for binary classification problems.

The implementation uses the `ndarray` crate for efficient numerical operations on arrays.

## Formula

The binary cross-entropy loss is calculated using the following formula:

```
CE = - Σ [ yᵢ * ln(pᵢ) + (1 - yᵢ) * ln(1 - pᵢ) ]
```

Where:

*   `yᵢ` is the true binary label for sample `i` (typically 0.0 or 1.0).
*   `pᵢ` is the predicted probability for sample `i` belonging to the positive class (label 1.0). This value should be between 0.0 and 1.0.
*   `ln` denotes the natural logarithm (base *e*).
*   `Σ` denotes the summation over all samples `i`.

The function returns the **total** cross-entropy loss summed over all samples, not the average loss.

## Implementation Details

*   **Natural Logarithm:** Uses the natural logarithm (`f32::ln`) as is standard for cross-entropy.
*   **Numerical Stability:** Adds a small constant `EPSILON` (1e-15) inside the logarithm calculations (`ln(p + EPSILON)` and `ln(1 - p + EPSILON)`). This prevents errors caused by taking the logarithm of exactly 0.0 or 1.0, which would result in negative infinity.
*   **`ndarray`:** Leverages `ndarray` for element-wise operations and summation, ensuring efficient computation.

## Usage

### Dependencies

Add `ndarray` to your `Cargo.toml`:

```toml
[dependencies]
ndarray = "0.15" # Use the latest compatible version
```

### Function Signature

```rust
use ndarray::Array1;

const EPSILON: f32 = 1e-15; // Or import std::f32::EPSILON

/// Calculates the binary cross-entropy loss.
///
/// Formula: CE = - Σ [ yᵢ * ln(pᵢ + ε) + (1 - yᵢ) * ln(1 - pᵢ + ε) ]
/// where ε is a small epsilon for numerical stability.
///
/// # Arguments
///
/// * `y` - A 1-D array (`&Array1<f32>`) of true binary labels (0.0 or 1.0).
/// * `p` - A 1-D array (`&Array1<f32>`) of predicted probabilities (between 0.0 and 1.0).
///         Must have the same length as `y`.
///
/// # Returns
///
/// * `f32` - The total cross-entropy loss summed over all samples.
///
/// # Panics
///
/// * Panics if `y` and `p` do not have the same length.
fn cross_entropy(y: &Array1<f32>, p: &Array1<f32>) -> f32 {
    // Ensure inputs have the same length
    assert_eq!(y.len(), p.len(), "Input arrays Y and P must have the same length.");

    // Calculate the first term: y * ln(p + ε)
    let term1 = y * &(p + EPSILON).mapv(f32::ln);

    // Calculate the second term: (1 - y) * ln(1 - p + ε)
    let term2 = (1.0 - y) * &(1.0 - p + EPSILON).mapv(f32::ln);

    // Sum the terms element-wise, sum all elements, and negate.
    - (term1 + term2).sum()
}
```

*(Note: The actual function definition might reside in a separate `.rs` file, e.g., `src/lib.rs` or `src/main.rs`)*

### Example

```rust
use ndarray::array;
use ndarray::Array1; // Make sure Array1 is in scope if needed directly

// Assuming the cross_entropy function definition is accessible
// (e.g., defined in the same file or imported from a library)

// Define EPSILON if not already defined/imported
const EPSILON: f32 = 1e-15;

fn cross_entropy(y: &Array1<f32>, p: &Array1<f32>) -> f32 {
    assert_eq!(y.len(), p.len(), "Input arrays Y and P must have the same length.");
    let term1 = y * &(p + EPSILON).mapv(f32::ln);
    let term2 = (1.0 - y) * &(1.0 - p + EPSILON).mapv(f32::ln);
    - (term1 + term2).sum()
}


fn main() {
    // Example usage
    let y_true = array![1.0, 0.0, 1.0, 1.0];      // True labels
    let p_pred = array![0.9, 0.1, 0.8, 0.95]; // Predicted probabilities

    let ce_loss = cross_entropy(&y_true, &p_pred);

    println!("True Labels: {:?}", y_true);
    println!("Predicted Probabilities: {:?}", p_pred);
    println!("Binary Cross-Entropy Loss: {}", ce_loss);
    // Expected output will be around 0.484 (sum of individual losses)
}
```

### Running the Example

1.  Save the example code as `src/main.rs`.
2.  Ensure your `Cargo.toml` includes `ndarray`.
3.  Run the code using Cargo:
    ```bash
    cargo run
    ```

```