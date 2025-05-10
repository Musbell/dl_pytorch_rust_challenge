use burn::tensor::{backend::Backend, Tensor, Distribution};
use burn::backend::Wgpu;
use burn::nn::Sigmoid; // Import the Sigmoid layer

type MyBackend = Wgpu;

fn main() {


    MyBackend::seed(7); // Set the random seed so things are predictable

    let device = Default::default();

    // Features: single sample with 5 features, shape [batch_size, num_features] = [1, 5]
    let features = Tensor::<MyBackend,2>::random([1, 5], Distribution::Default, &device);

    // Weights: maps 5 features to 1 output, shape [num_features, num_output] = [5, 1]
    let weights = Tensor::<MyBackend,2>::random([5, 1], Distribution::Default, &device);

    // Bias: per output neuron, shape [batch_size, num_output] = [1, 1]
    let bias = Tensor::<MyBackend,2>::random([1, 1], Distribution::Default, &device);

    // --- Linear Calculation ---
    // Calculate the linear output: (features [1,5] @ weights [5,1]) + bias [1,1] -> [1,1] tensor
    let linear_output = features.matmul(weights) + bias;

    // --- Activation using burn::nn::Sigmoid ---
    // Create an instance of the Sigmoid module
    let sigmoid = Sigmoid::new();

    // Apply the sigmoid activation to the linear output tensor
    let sigmoid_output_tensor = sigmoid.forward(linear_output);

    // Get the single scalar value from the [1, 1] tensor
    // Sigmoid::forward returns a Tensor<B, D>, we extract the scalar
    let output_f32: f32 = sigmoid_output_tensor.into_scalar();

    println!("output_f32: {:?}", output_f32);
}