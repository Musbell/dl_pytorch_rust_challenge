use ndarray::prelude::*;
use ndarray_stats::QuantileExt;


// Write a function that takes as input a list of numbers, and returns
// the list of values given by the softmax function.
fn softmax_stable(l: &Array1<f64>) -> Array1<f64> {
    if l.is_empty() {
        return Array::zeros(l.len());
    }

    let max_val = *l.max().expect("Failed to find maximum value (NaNs?)");

    let exp_l_shifted = (l - max_val).mapv(f64::exp);
    let sum_exp_shifted = exp_l_shifted.sum();
    
    let softmax_output = exp_l_shifted / sum_exp_shifted;
    
    softmax_output
}

fn main() {
    // Test the stable softmax function
    let l_stable = array![1.0, 2.0, 3.0];
    let result_stable = softmax_stable(&l_stable);
    println!("Stable Softmax (normal): {:?}", result_stable);

    // Check that the sum of the softmax values is 1
    let sum: f64 = result_stable.sum();
    println!("Sum (normal): {}", sum);
    assert!((sum - 1.0).abs() < 1e-10, "Softmax values do not sum to 1");

    println!("---");

    // Test with large numbers
    let l_large = array![900.0, 901.0, 902.0];
    let result_large = softmax_stable(&l_large);
    println!("Stable Softmax (large): {:?}", result_large);
    // Expected output approx: [0.09003057, 0.24472847, 0.66524096]
    let sum_large: f64 = result_large.sum();
    println!("Sum (large): {}", sum_large);
    assert!((sum_large - 1.0).abs() < 1e-10, "Softmax values do not sum to 1");


    // Test with small numbers (where original might underflow)
    let l_small = array![-900.0, -901.0, -902.0];
    let result_small = softmax_stable(&l_small);
    println!("Stable Softmax (small): {:?}", result_small);
    // Expected output approx: [0.66524096, 0.24472847, 0.09003057]
    let sum_small: f64 = result_small.sum();
    println!("Sum (small): {}", sum_small);
    assert!((sum_small - 1.0).abs() < 1e-10, "Softmax values do not sum to 1");
}
