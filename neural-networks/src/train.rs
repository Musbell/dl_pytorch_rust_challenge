use burn::optim::{GradientsParams, Optimizer, SgdConfig};
use burn::prelude::*;
use burn::record::{DefaultFileRecorder, FullPrecisionSettings};
use burn_ndarray::{NdArray, NdArrayDevice};
use std::error::Error;

use crate::model::{MnistTrainingConfig, Network, NetworkConfig, build_dataloaders};

type AutodiffNdArray = burn::backend::Autodiff<NdArray>;

fn create_artifact_dir(artifact_dir: &str) {
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn train(
    device: &NdArrayDevice,
    epochs: usize,
    learning_rate: f64,
) -> Result<Network<AutodiffNdArray>, Box<dyn Error>> {
    let config = MnistTrainingConfig::new();
    let (dataloader_train, _dataloader_test) = build_dataloaders(&config);

    // Build model
    let net_cfg = NetworkConfig::new();
    let mut model: Network<AutodiffNdArray> = net_cfg.init(device);

    // Initialize optimizer
    let mut optimizer = SgdConfig::new().init();

    println!(
        "Number of train batches: {}",
        dataloader_train.iter().count()
    );

    // Training loop
    for epoch in 0..epochs {
        let mut running_loss = 0.0;
        let mut batch_count = 0;

        for batch in dataloader_train.iter() {
            // Convert tensors to autodiff backend
            let images: Tensor<AutodiffNdArray, 3> =
                Tensor::from_data(batch.images.to_data(), device);
            let targets: Tensor<AutodiffNdArray, 1, Int> =
                Tensor::from_data(batch.targets.to_data(), device);

            // Flatten MNIST images into a 784 long vector
            let dims = images.dims();
            let batch_size = dims[0];
            let num_features = dims[1] * dims[2];
            let images = images.reshape([batch_size, num_features]);

            // Forward pass
            let output = model.forward(images);
            let loss = model.loss(output, targets, device);

            // Backward pass and optimizer step
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optimizer.step(learning_rate, model, grads);

            // Track loss
            let loss_value: f32 = loss.into_scalar();
            running_loss += loss_value;
            batch_count += 1;
        }

        // Print average loss for this epoch
        let avg_loss = running_loss / batch_count as f32;
        println!("Epoch {}: Training loss: {}", epoch + 1, avg_loss);
    }

    // Return the trained model
    Ok(model)
}

/// Convert autodiff model to inference model (removes gradient tracking)
pub fn to_inference_model(
    model: Network<AutodiffNdArray>,
    device: &NdArrayDevice,
) -> Network<NdArray> {
    // Save the trained model to a temporary file
    let temp_path = "/tmp/burn-example-mnist/temp_model";
    let recorder = DefaultFileRecorder::<FullPrecisionSettings>::new();
    model
        .save_file(temp_path, &recorder)
        .expect("Failed to save model");

    // Create a new inference model and load the weights
    let net_cfg = NetworkConfig::new();
    let inference_model = net_cfg.init::<NdArray>(device);

    // Load the trained weights
    inference_model
        .load_file(temp_path, &recorder, device)
        .expect("Failed to load model")
}

/// Make predictions on test data
pub fn evaluate_model(
    model: &Network<NdArray>,
    device: &NdArrayDevice,
) -> Result<(), Box<dyn Error>> {
    let config = MnistTrainingConfig::new();
    let (_dataloader_train, dataloader_test) = build_dataloaders(&config);

    let mut correct = 0;
    let mut total = 0;
    let mut examples_shown = 0;

    println!(
        "
=== Model Evaluation ==="
    );

    for batch in dataloader_test.iter() {
        // Convert to inference backend
        let images: Tensor<NdArray, 3> = Tensor::from_data(batch.images.to_data(), device);
        let targets: Tensor<NdArray, 1, Int> = Tensor::from_data(batch.targets.to_data(), device);

        // Flatten images for model input
        let dims = images.dims();
        let batch_size = dims[0];
        let num_features = dims[1] * dims[2];
        let images = images.reshape([batch_size, num_features]);

        // Make predictions (log probabilities)
        let log_probs = model.forward(images);

        // Convert log probabilities to actual probabilities
        let probs = log_probs.exp();

        // Get predicted classes (argmax)
        let predictions = probs.clone().argmax(1);

        // Calculate accuracy
        let targets_data: Vec<i64> = targets.to_data().into_vec().unwrap();
        let predictions_data: Vec<i64> = predictions.to_data().into_vec().unwrap();

        for i in 0..batch_size {
            let predicted = predictions_data[i];
            let actual = targets_data[i];

            if predicted == actual {
                correct += 1;
            }
            total += 1;

            // Show first 10 examples
            if examples_shown < 10 {
                let single_probs = probs.clone().slice([i..i + 1, 0..10]);
                let probs_data: Vec<f32> = single_probs.to_data().into_vec().unwrap();
                println!(
                    "Example {}: Predicted = {}, Actual = {}, Confidence = {:.3}",
                    examples_shown + 1,
                    predicted,
                    actual,
                    probs_data[predicted as usize]
                );
                examples_shown += 1;
            }
        }

        // Only process first batch for demo
        break;
    }

    let accuracy = correct as f32 / total as f32 * 100.0;
    println!(
        "
Accuracy: {}/{} ({:.2}%)",
        correct, total, accuracy
    );

    Ok(())
}

/// Show detailed prediction for first test image
pub fn show_prediction_details(
    model: &Network<NdArray>,
    device: &NdArrayDevice,
) -> Result<(), Box<dyn Error>> {
    let config = MnistTrainingConfig::new();
    let (_dataloader_train, dataloader_test) = build_dataloaders(&config);

    // Get first batch
    let batch = dataloader_test.iter().next().unwrap();
    let images: Tensor<NdArray, 3> = Tensor::from_data(batch.images.to_data(), device);
    let targets: Tensor<NdArray, 1, Int> = Tensor::from_data(batch.targets.to_data(), device);

    // Process first image only
    let single_image = images.slice([0..1, 0..28, 0..28]).reshape([1, 784]);
    let single_target = targets.slice([0..1]);

    let log_probs = model.forward(single_image);
    let probs = log_probs.exp();

    let probs_data: Vec<f32> = probs.to_data().into_vec().unwrap();
    let actual_label: i64 = single_target.to_data().into_vec().unwrap()[0];

    println!(
        "
=== Detailed Prediction for First Test Image ==="
    );
    println!("Actual label: {}", actual_label);
    println!("Predicted probabilities for each digit:");

    for (digit, &prob) in probs_data.iter().enumerate() {
        let marker = if digit == actual_label as usize {
            " <- TRUE"
        } else {
            ""
        };
        println!(
            "  Digit {}: {:.4} ({:.1}%){}",
            digit,
            prob,
            prob * 100.0,
            marker
        );
    }

    let predicted_digit = probs_data
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap()
        .0;

    println!("Predicted digit: {}", predicted_digit);
    println!("Confidence: {:.2}%", probs_data[predicted_digit] * 100.0);

    Ok(())
}

/// Main training and evaluation pipeline
pub fn run() -> Result<(), Box<dyn Error>> {
    create_artifact_dir("/tmp/burn-example-mnist");

    // Training parameters
    let epochs = 10;
    let learning_rate = 0.003;

    // Use autodiff backend for training
    let device = NdArrayDevice::default();

    // Train the model
    println!("=== Training Phase ===");
    let trained_model = train(&device, epochs, learning_rate)?;

    // Convert to inference model and evaluate
    println!(
        "
=== Converting to Inference Model ==="
    );
    let inference_model = to_inference_model(trained_model, &device);

    // Evaluate the model
    evaluate_model(&inference_model, &device)?;

    // Show detailed predictions
    show_prediction_details(&inference_model, &device)?;

    Ok(())
}
