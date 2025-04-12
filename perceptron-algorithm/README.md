# Perceptron Algorithm in Rust

## Overview
This Rust project implements a simple perceptron algorithm designed for binary classification tasks. It leverages CSV data as input, performs iterative training to adjust weights, and visualizes the decision boundary evolution through epochs.

## Project Structure
- `src/main.rs`: Core Rust implementation, including CSV parsing, perceptron training, and visualization.
- `data.csv`: Input data file (ensure this is located in the crate root).
- `perceptron_plot.png`: Output visualization illustrating the classification and decision boundaries.

## Key Dependencies
- **ndarray**: Efficient numerical arrays for Rust.
- **ndarray-rand**: Random number generation for initializing weights.
- **csv**: Reading CSV data files.
- **plotters**: Plotting decision boundaries and data points.
- **rand, rand_chacha**: Reproducible random number generation.

## Features
- **Data Parsing**: Robust CSV parsing with feature validation.
- **Training**: Adjustable learning rate, epochs, and reproducible training via random seed.
- **Visualization**: Real-time visual updates of perceptron decision boundaries after training.

## How to Run

### Requirements
- Rust installed (recommended version: Rust 1.75+)

### Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Musbell/dl_pytorch_rust_challenge.git
   cd dl_pytorch_rust_challenge/perceptron-algorithm
   ```

2. **Run the perceptron algorithm**:
   ```bash
   cargo run --release
   ```

3. **Check Results**:
    - Terminal will display training details and boundary lines.
    - A plot named `perceptron_plot.png` will be generated in your directory.

## Configuration
To customize training parameters, edit `src/main.rs`:

```rust
let learning_rate = 0.01;
let epochs = 25;
let seed: u64 = 42;
```

## Output
After training completes, you will get:
- Terminal output summarizing epochs, final weights, and biases.
- A visual plot (`perceptron_plot.png`) showing class separation and decision boundary progress.


