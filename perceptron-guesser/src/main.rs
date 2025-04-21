use polars::prelude::*;
use rand::Rng;
use std::error::Error;
use std::io::{self, Write};

// --- Constants ---
const BOOLEAN_INPUTS: [(i32, i32); 4] = [(0, 0), (0, 1), (1, 0), (1, 1)];
const MAX_ATTEMPTS: usize = 3;
const WEIGHT_RANGE: std::ops::Range<f64> = -2.0..2.0;

// --- Standard Gate Patterns ---
const AND_PATTERN: [bool; 4] = [false, false, false, true];
const OR_PATTERN: [bool; 4] = [false, true, true, true];
// Add more patterns here if desired (e.g., NAND, NOR)
// const NAND_PATTERN: [bool; 4] = [true, true, true, false];
// const NOR_PATTERN: [bool; 4] = [true, false, false, false];

// --- Perceptron & EvaluationResult (Remain the same) ---
struct Perceptron {
    weight1: f64,
    weight2: f64,
    bias: f64,
}

struct EvaluationResult {
    dataframe: DataFrame,
    wrong_count: usize,
}

impl Perceptron {
    fn new(weight1: f64, weight2: f64, bias: f64) -> Self {
        Perceptron { weight1, weight2, bias }
    }

    fn evaluate(&self, expected_outputs: &[bool]) -> Result<EvaluationResult, PolarsError> {
        if expected_outputs.len() != BOOLEAN_INPUTS.len() {
            return Err(PolarsError::ComputeError(format!(
                "Mismatch between the number of inputs ({}) and expected outputs ({})",
                BOOLEAN_INPUTS.len(), expected_outputs.len()
            ).into()));
        }

        let n_inputs = BOOLEAN_INPUTS.len();
        let mut in1_col = Vec::with_capacity(n_inputs);
        let mut in2_col = Vec::with_capacity(n_inputs);
        let mut lin_col = Vec::with_capacity(n_inputs);
        let mut act_col = Vec::with_capacity(n_inputs);
        let mut corr_col = Vec::with_capacity(n_inputs);
        let mut wrong_count = 0;

        for (&(input1, input2), &expected) in BOOLEAN_INPUTS.iter().zip(expected_outputs) {
            let linear_combination = self.weight1 * (input1 as f64)
                + self.weight2 * (input2 as f64)
                + self.bias;
            let prediction = linear_combination >= 0.0;

            in1_col.push(input1);
            in2_col.push(input2);
            lin_col.push(format!("{:.1}", linear_combination));
            act_col.push(prediction as i32);

            let is_correct = prediction == expected;
            corr_col.push(if is_correct { "Yes" } else { "No" });

            if !is_correct {
                wrong_count += 1;
            }
        }

        let df = DataFrame::new(vec![
            Series::new(PlSmallStr::from("Input 1"), in1_col).into_column(),
            Series::new(PlSmallStr::from("Input 2"), in2_col).into_column(),
            Series::new(PlSmallStr::from("Linear Combination"), lin_col).into_column(),
            Series::new(PlSmallStr::from("Activation Output"), act_col).into_column(),
            Series::new(PlSmallStr::from("Is Correct"), corr_col).into_column(),
        ])?;

        Ok(EvaluationResult { dataframe: df, wrong_count })
    }
}

// --- Helper function to get user input (Remains the same) ---
fn prompt_and_read_f64(prompt: &str) -> Result<f64, io::Error> {
    loop {
        print!("{}", prompt);
        io::stdout().flush()?;
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        match input.trim().parse::<f64>() {
            Ok(value) => return Ok(value),
            Err(_) => println!("Invalid input. Please enter a number (e.g., 1.0, -1.5)."),
        }
    }
}

// --- Function to Generate Target Pattern (Remains the same) ---
fn generate_target_pattern() -> Vec<bool> {
    let mut rng = rand::rng();
    let w1_secret = rng.random_range(WEIGHT_RANGE);
    let w2_secret = rng.random_range(WEIGHT_RANGE);
    let b_secret = rng.random_range(WEIGHT_RANGE);

    let target_pattern: Vec<bool> = BOOLEAN_INPUTS
        .iter()
        .map(|&(i1, i2)| {
            let linear_combination = w1_secret * (i1 as f64)
                + w2_secret * (i2 as f64)
                + b_secret;
            linear_combination >= 0.0
        })
        .collect();

    // println!(// Optional Hint
    //     "Hint: Secret params used: w1={:.2}, w2={:.2}, b={:.2}",
    //     w1_secret, w2_secret, b_secret
    // );

    target_pattern
}

// --- NEW Function to Identify the Pattern Type ---
/// Compares the generated pattern against known gate patterns.
fn identify_pattern_type(pattern: &[bool]) -> String {
    if pattern == AND_PATTERN {
        "the AND Gate".to_string()
    } else if pattern == OR_PATTERN {
        "the OR Gate".to_string()
    // } else if pattern == NAND_PATTERN { // Example if adding more
    //     "the NAND Gate".to_string()
    // } else if pattern == NOR_PATTERN {
    //     "the NOR Gate".to_string()
    } else {
        // Check for all true/false explicitly if desired
        if pattern.iter().all(|&x| x) {
            "a TAUTOLOGY (all True)".to_string()
        } else if pattern.iter().all(|&x| !x) {
            "a CONTRADICTION (all False)".to_string() // Like the example you got!
        } else {
            "a Custom Pattern".to_string()
        }
    }
}


// --- Game Logic ---

fn run_game() -> Result<(), Box<dyn Error>> {
    println!("--- Welcome to the Perceptron Parameter Guessing Game! ---");
    println!("A random, solvable target output pattern will be generated.");
    println!("Try to find weights (w1, w2) and bias (b) that match the target pattern.");

    // Generate the target pattern
    let target_outputs = generate_target_pattern();
    // **Identify the pattern type**
    let pattern_type = identify_pattern_type(&target_outputs);

    println!("\nYour target: Match this output pattern!");
    // **Tell the user the identified type**
    println!("-> The generated target pattern corresponds to **{}**!", pattern_type);
    println!("Inputs: {:?}", BOOLEAN_INPUTS);
    let target_display: Vec<i32> = target_outputs.iter().map(|&b| b as i32).collect();
    println!("Target: {:?}", target_display);
    println!("You have {} attempts.", MAX_ATTEMPTS);


    let mut attempts = 0;
    let mut won = false;

    while attempts < MAX_ATTEMPTS {
        attempts += 1;
        println!("\n--- Attempt {} of {} ---", attempts, MAX_ATTEMPTS);

        let w1 = prompt_and_read_f64("Enter your guess for weight 1 (w1): ")?;
        let w2 = prompt_and_read_f64("Enter your guess for weight 2 (w2): ")?;
        let b = prompt_and_read_f64("Enter your guess for bias (b): ")?;

        let perceptron = Perceptron::new(w1, w2, b);
        println!("Evaluating Perceptron with w1={}, w2={}, b={}", w1, w2, b);

        match perceptron.evaluate(&target_outputs) {
            Ok(results) => {
                println!("\nEvaluation Result for Attempt {}:", attempts);
                println!("{}", results.dataframe);

                if results.wrong_count == 0 {
                    println!("\n*********************************************");
                    // **Update win message**
                    println!("*** Congratulations! You matched the target pattern for {}! ***", pattern_type);
                    println!("*********************************************");
                    won = true;
                    break;
                } else {
                    println!("\nIncorrect. {} predictions were wrong.", results.wrong_count);
                    if attempts < MAX_ATTEMPTS {
                        println!("You have {} attempts left.", MAX_ATTEMPTS - attempts);
                    }
                }
            }
            Err(e) => {
                eprintln!("\nError during evaluation: {}. Please try again.", e);
            }
        }
    }

    if !won {
        println!("\n---------------------------------------------");
        println!("--- Game Over! You ran out of attempts. ---");
        // **Update lose message**
        println!("--- The target pattern ({}) was {:?} ---", pattern_type, target_display);
        println!("---------------------------------------------");
    }

    Ok(())
}

// --- Main Execution ---

fn main() {
    if let Err(e) = run_game() {
        eprintln!("An error occurred: {}", e);
    }
}