# Perceptron Parameter Guessing Game

Test your intuition about basic machine learning concepts! This is a simple command-line game written in Rust where you try to guess the parameters (weights and bias) of a single perceptron to match a randomly generated, linearly separable output pattern.

## Features

* Generates a random target output pattern that is guaranteed to be solvable by a single perceptron.
* Identifies if the random pattern matches standard gates like AND or OR, or other simple patterns (all True/False).
* Provides detailed feedback after each guess using a table (via the Polars library).
* Gives you 3 attempts to find the correct parameters.

## Prerequisites

* **Rust Toolchain:** You need `rustc` and `cargo` installed. If you don't have them, visit [https://rustup.rs/](https://rustup.rs/) to install them easily.

## Building the Game

Run the build command:
```bash
cargo build --release
```

## Running the Game

After building, run the game with the following command:
```bash
cargo run --release
```

## How to Play

1. When you run the game, it welcomes you and generates a random target output pattern for the standard boolean inputs: `[(0, 0), (0, 1), (1, 0), (1, 1)]`.
2. The game identifies and informs you about the generated pattern (e.g., "the AND Gate", "a CONTRADICTION (all False)", "a Custom Pattern").
3. It displays the four standard `Inputs` and the `Target` output pattern (represented as `0` for False and `1` for True) that your perceptron needs to replicate.
4. You are prompted to enter your guess for:
    * `weight 1 (w1)`
    * `weight 2 (w2)`
    * `bias (b)`
5. Enter numerical values (like `1.0`, `-0.5`, `0`, `1`, `-2.2`) for each parameter when prompted.
6. After each attempt (you have 3 in total), the game evaluates a perceptron using your guessed parameters.
7. It prints a **results table**:
    * `Input 1`, `Input 2`: The standard input pair.
    * `Linear Combination`: The calculated value `w1*Input1 + w2*Input2 + b`.
    * `Activation Output`: The perceptron's output (1 if Linear Combination ≥ 0, else 0).
    * `Is Correct`: 'Yes' if the Activation Output matches the Target for that row, 'No' otherwise.
8. **Your Goal:** Find values for `w1`, `w2`, and `b` that make the 'Is Correct' column show 'Yes' for all four input rows.
9. If you succeed within 3 attempts, you win! If you run out of attempts, the game ends, and it reveals the target pattern again.

Good luck!

