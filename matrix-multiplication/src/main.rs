use std::default::Default;
use burn::tensor::{backend::Backend, Tensor, Distribution};
use burn::backend::Wgpu;
use burn::nn::Sigmoid; // Import the Sigmoid layer

// Define the backend to use
type MyBackend = Wgpu;

fn main() {
    // Set the random seed for reproducible results
    MyBackend::seed(7);

    // Get the default device for the chosen backend (WGPU device)
    let device = Default::default();

    // --- Define Network Dimensions ---
    // Input features: 1 sample with 3 features
    let n_input = 3;
    // Hidden layer: 2 neurons
    let n_hidden = 2;
    // Output layer: 1 neuron
    let n_output = 1;
    let batch_size = 1;

    // --- Initialize Tensors ---
    // Input features: shape [batch_size, n_input] = [1, 3]
    let features = Tensor::<MyBackend, 2>::random(
        [batch_size, n_input],
        Distribution::Default,
        &device);

    // Weights for the first layer (input to hidden): shape [n_input, n_hidden] = [3, 2]
    let w_1 = Tensor::<MyBackend, 2>::random(
        [n_input, n_hidden],
        Distribution::Default,
        &device);

    // Weights for the second layer (hidden to output): shape [n_hidden, n_output] = [2, 1]
    let w_2 = Tensor::<MyBackend, 2>::random(
        [n_hidden, n_output],
        Distribution::Default,
        &device);

    // Bias for the first layer (hidden layer): shape [batch_size, n_hidden] = [1, 2]
    let b_1 = Tensor::<MyBackend, 2>::random(
        [batch_size, n_hidden],
        Distribution::Default,
        &device);

    // Bias for the second layer (output layer): shape [batch_size, n_output] = [1, 1]
    let b_2 = Tensor::<MyBackend, 2>::random(
        [batch_size, n_output],
        Distribution::Default,
        &device);

    // --- Forward Pass Calculation ---

    //  Calculate the linear output of the hidden layer
    // (features [1,3] @ w_1 [3,2]) + b_1 [1,2] -> result shape [1,2]
    let linear_h = features.matmul(w_1) + b_1;

    // Apply Activation to the hidden layer output
    // Create an instance of the Sigmoid module
    let sigmoid = Sigmoid::new();
    // Apply sigmoid element-wise to the linear hidden output -> result shape [1,2]
    let h_activated = sigmoid.forward(linear_h); 

    // Calculate the linear output of the output layer
    // (h_activated [1,2] @ w_2 [2,1]) + b_2 [1,1] -> result shape [1,1]
    let final_linear_output = h_activated.matmul(w_2) + b_2;

    // Apply a final activation
    let final_output_with_sigmoid = sigmoid.forward(final_linear_output);

    // --- Output Result --- :
    println!("Final output after sigmoid: {:?}", final_output_with_sigmoid.into_scalar());
}