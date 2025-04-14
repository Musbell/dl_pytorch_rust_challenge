# Numerically Stable Softmax in Rust

This Rust project implements a numerically stable softmax function for 1-dimensional arrays (`ndarray::Array1<f64>`). It effectively addresses numerical issues commonly encountered in naive softmax implementations.

## What is Softmax?

The softmax function transforms a vector of real numbers into probabilities:

```math
softmax(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}
```

However, the naive calculation (`exp(x_i) / sum(exp(x_j))`) can encounter numerical instability:

- **Overflow**: Large positive numbers result in `exp(x_i)` being infinite.
- **Underflow**: Large negative numbers make `exp(x_i)` approach zero.

## Numerically Stable Implementation

This implementation mitigates instability by shifting input values by subtracting the maximum value before exponentiation:

```math
softmax(x_i) = \frac{e^{(x_i - max(x))}}{\sum_j e^{(x_j - max(x))}}
```

This method prevents overflow and minimizes underflow risks without altering the mathematical outcome.

## Dependencies

This project uses these crates:

- `ndarray` – for array manipulation
- `ndarray-stats` – for statistical operations

Add them to your `Cargo.toml`:

```toml
[dependencies]
ndarray = "0.16.1"
ndarray-stats = "0.6.0"
```

## Usage

Here's how to use the `softmax_stable` function:

```rust
use ndarray::prelude::*;
use ndarray_stats::QuantileExt;

fn softmax_stable(l: &Array1<f64>) -> Array1<f64> {
    if l.is_empty() {
        return Array::zeros(l.len());
    }

    let max_val = *l.max().expect("Failed to find maximum value (NaNs?)");
    let exp_l_shifted = (l - max_val).mapv(f64::exp);
    let sum_exp_shifted = exp_l_shifted.sum();

    exp_l_shifted / sum_exp_shifted
}

fn main() {
    let input = array![1.0, 0.5, 2.0, -1.0];
    let probabilities = softmax_stable(&input);

    println!("Input: {:?}", input);
    println!("Softmax Output: {:?}", probabilities);
    println!("Sum of probabilities: {}", probabilities.sum()); // ~1.0
}
```

## Running the Example

Ensure Rust and Cargo are installed:

```bash
# Compile and run the program
cargo run

# Compile and run optimized release version
cargo run --release
```

The output demonstrates numerical stability with inputs of varying magnitudes.

## Project Structure

Your project should have the following structure:

```
softmax/
├── Cargo.toml
└── src/
    └── main.rs
```

Example `Cargo.toml` configuration:

```toml
[package]
name = "softmax"
version = "0.1.0"
edition = "2021"

[dependencies]
ndarray = "0.16.1"
ndarray-stats = "0.6.0"
```

## Testing the Implementation

The provided `main.rs` includes test cases for normal, large, and small input values to verify correctness and stability.

Run tests using:

```bash
cargo run
```

Assertions confirm the sum of softmax outputs is approximately 1, validating numerical stability.

