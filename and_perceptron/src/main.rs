/**
What are the weights and bias for the AND perceptron?
Set the weights (WEIGHT_1, WEIGHT_2) and BIAS (bias) to values that will correctly determine the AND operation as shown above.
More than one set of values will work!
**/

use polars::prelude::*;

// TODO: Set WEIGHT_1, WEIGHT_2, and BIAS
fn main() {
    const WEIGHT_1: i32 = 1;
    const WEIGHT_2: i32 = 1;
    const BIAS: f64 =  -1.5;

    // Inputs and outputs
    let test_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)];
    let correct_outputs = [false, false, false, true];
    let mut outputs = Vec::new();

    // Generate and check output
    for (test_inputs, correct_outputs) in test_inputs.iter().zip(correct_outputs.iter()) {
        let linear_combination = (WEIGHT_1 as f64) * (test_inputs.0 as f64)
            + (WEIGHT_2 as f64) * (test_inputs.1 as f64)
            + BIAS;
        let output = linear_combination >= 0.0;
        let is_correct_string = if output == *correct_outputs {
            "Yes"
        } else {
            "No"
        };

        outputs.push((test_inputs, linear_combination, output, is_correct_string));
    }

    // Print output
    let num_wrong = outputs.iter().filter(|output| output.3 == "No").count();
    let output_frame : DataFrame = df!(
        "Input 1" => outputs.iter().map(|output| output.0.0).collect::<Vec<_>>(),
        "Input 2" => outputs.iter().map(|output| output.0.1).collect::<Vec<_>>(),
        "Linear Combination" => outputs.iter().map(|output| output.1).collect::<Vec<_>>(),
        "Activation Output" => outputs.iter().map(|output| output.2).collect::<Vec<_>>(),
        "Is Correct" => outputs.iter().map(|output| output.3).collect::<Vec<_>>()
    ).unwrap();

    if num_wrong == 0 {
        println!("Nice!  You got it all correct.\n");
    } else {
        println!("You got {} wrong.  Keep trying!\n", num_wrong);
    }

    println!("{}", output_frame);
}
