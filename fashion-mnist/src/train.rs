use burn::optim::{GradientsParams, Optimizer, SgdConfig};
use burn::prelude::*;
use burn::record::{CompactRecorder, DefaultFileRecorder, FullPrecisionSettings, Recorder};
use burn::backend::Wgpu;
use burn::data::dataset::vision::MnistItem;
use burn::data::dataloader::batcher::Batcher;
use std::error::Error;

use crate::data::FashionMNISTBatcher;
use crate::dataset::FashionMNISTDataset;
use crate::model::{MnistTrainingConfig, Network, NetworkConfig, build_dataloaders};
use crate::visualization;

// Fashion MNIST class names
pub fn get_class_name(label: i32) -> &'static str {
    match label {
        0 => "T-shirt/top",
        1 => "Trouser",
        2 => "Pullover",
        3 => "Dress",
        4 => "Coat",
        5 => "Sandal",
        6 => "Shirt",
        7 => "Sneaker",
        8 => "Bag",
        9 => "Ankle boot",
        _ => "Unknown",
    }
}

type AutodiffWgpu = burn::backend::Autodiff<Wgpu>;

fn create_artifact_dir(artifact_dir: &str) {
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

/// Train a model
pub fn train(
    device: &<Wgpu as Backend>::Device,
    epochs: usize,
    learning_rate: f64,
) -> Result<(Network<AutodiffWgpu>, Vec<f32>), Box<dyn Error>> {
    let config = MnistTrainingConfig::new();
    let (dataloader_train, _dataloader_test) = build_dataloaders(&config);

    // Build model
    let net_cfg = NetworkConfig::new();
    let mut model: Network<AutodiffWgpu> = net_cfg.init(device);

    // Initialize optimizer
    let mut optimizer = SgdConfig::new().init();

    println!(
        "Number of train batches: {}",
        dataloader_train.iter().count()
    );

    // Track losses for visualization
    let mut epoch_losses = Vec::with_capacity(epochs);

    // Training loop
    for epoch in 0..epochs {
        let mut running_loss = 0.0;
        let mut batch_count = 0;

        for batch in dataloader_train.iter() {
            // Convert tensors to autodiff backend
            let images: Tensor<AutodiffWgpu, 3> =
                Tensor::from_data(batch.images.to_data(), device);
            let targets: Tensor<AutodiffWgpu, 1, Int> =
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

        // Store the average loss for visualization
        epoch_losses.push(avg_loss);
    }

    // Return the trained model and losses
    Ok((model, epoch_losses))
}


/// Convert autodiff model to inference model (removes gradient tracking)
pub fn to_inference_model(
    model: Network<AutodiffWgpu>,
    device: &<Wgpu as Backend>::Device,
) -> Network<Wgpu> {
    // Save the trained model to a temporary file
    let temp_path = "/tmp/burn-example-mnist/temp_model_wgpu";
    let recorder = DefaultFileRecorder::<FullPrecisionSettings>::new();
    model
        .save_file(temp_path, &recorder)
        .expect("Failed to save model");

    // Create a new inference model and load the weights
    let net_cfg = NetworkConfig::new();
    let inference_model = net_cfg.init::<Wgpu>(device);

    // Load the trained weights
    inference_model
        .load_file(temp_path, &recorder, device)
        .expect("Failed to load model")
}


/// Make predictions on test data
pub fn evaluate_model(
    model: &Network<Wgpu>,
    device: &<Wgpu as Backend>::Device,
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
        let images: Tensor<Wgpu, 3> = Tensor::from_data(batch.images.to_data(), device);
        let targets: Tensor<Wgpu, 1, Int> = Tensor::from_data(batch.targets.to_data(), device);

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
        let targets_data: Vec<i32> = targets.to_data().into_vec().unwrap();
        let predictions_data: Vec<i32> = predictions.to_data().into_vec().unwrap();

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
                    "Example {}: Predicted = {} ({}), Actual = {} ({}), Confidence = {:.3}",
                    examples_shown + 1,
                    get_class_name(predicted),
                    predicted,
                    get_class_name(actual),
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
    model: &Network<Wgpu>,
    device: &<Wgpu as Backend>::Device,
) -> Result<(), Box<dyn Error>> {
    let config = MnistTrainingConfig::new();
    let (_dataloader_train, dataloader_test) = build_dataloaders(&config);

    // Get first batch
    let batch = dataloader_test.iter().next().unwrap();
    let images: Tensor<Wgpu, 3> = Tensor::from_data(batch.images.to_data(), device);
    let targets: Tensor<Wgpu, 1, Int> = Tensor::from_data(batch.targets.to_data(), device);

    // Process first image only
    let single_image = images.slice([0..1, 0..28, 0..28]).reshape([1, 784]);
    let single_target = targets.slice([0..1]);

    let log_probs = model.forward(single_image);
    let probs = log_probs.exp();

    let probs_data: Vec<f32> = probs.to_data().into_vec().unwrap();
    let actual_label: i32 = single_target.to_data().into_vec().unwrap()[0];

    println!(
        "
=== Detailed Prediction for First Test Image ==="
    );
    println!("Actual label: {} ({})", get_class_name(actual_label), actual_label);
    println!("Predicted probabilities for each class:");

    for (digit, &prob) in probs_data.iter().enumerate() {
        let marker = if digit == actual_label as usize {
            " <- TRUE"
        } else {
            ""
        };
        println!(
            "  Class {} ({}): {:.4} ({:.1}%){}",
            get_class_name(digit as i32),
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

    println!("Predicted class: {} ({})", get_class_name(predicted_digit as i32), predicted_digit);
    println!("Confidence: {:.2}%", probs_data[predicted_digit] * 100.0);

    Ok(())
}

/// Infer a single item using a trained model
pub fn infer<B: Backend>(artifact_dir: &str, device: B::Device, item: MnistItem) {
    // Load the network configuration
    let net_cfg = NetworkConfig::new();

    // Initialize the model
    let model = net_cfg.init::<B>(&device);

    // Load the trained weights
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), &device)
        .expect("Trained model should exist");

    let model = model.load_record(record);

    // Get the label from the item
    let label = item.label;

    // Create a batcher and batch the item
    let batcher = FashionMNISTBatcher::default();
    let batch = batcher.batch(vec![item], &device);

    // Reshape the image for the model input
    let images = batch.images.reshape([1, 784]);

    // Forward pass
    let output = model.forward(images);

    // Get the predicted class
    let predictions = output.argmax(1);
    let predictions_data: Vec<i32> = predictions.to_data().into_vec().unwrap();
    let predicted = predictions_data[0];

    // Convert numeric labels to class names
    let predicted_class = get_class_name(predicted);
    let actual_class = get_class_name(label as i32);

    // Print the results
    println!("Predicted {} ({}) Expected {} ({})", 
             predicted_class, predicted, 
             actual_class, label);
}

/// Main training and evaluation pipeline
pub fn run() -> Result<(), Box<dyn Error>> {
    // Use WGPU backend
    create_artifact_dir("/tmp/burn-example-mnist-wgpu");

    // Training parameters
    let epochs = 15;
    let learning_rate = 0.003;

    // Use WGPU backend for training
    let device = Default::default();

    // Create dataset instances for visualization
    let train_dataset = FashionMNISTDataset::train();
    let test_dataset = FashionMNISTDataset::test();

    // Visualize sample images from the dataset
    visualization::visualize_image_grid(&train_dataset, "train_samples.png")?;

    // Train the model
    println!("=== Training Phase ===");
    let (trained_model, losses) = train(&device, epochs, learning_rate)?;

    // Visualize training losses
    visualization::visualize_training_loss(&losses, "training_loss.png")?;

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

    // Visualize model predictions
    visualization::visualize_predictions(&inference_model, &test_dataset, &device, "predictions.png")?;

    // Visualize confusion matrix
    visualization::visualize_confusion_matrix(&inference_model, &test_dataset, &device, "confusion_matrix.png")?;

    println!("
=== Visualizations ===
Visualizations have been saved to the ./fashion-mnist/visualizations directory:
- train_samples.png: Grid of training images
- training_loss.png: Plot of training loss over epochs
- predictions.png: Grid of test images with model predictions
- confusion_matrix.png: Confusion matrix of model predictions
");

    Ok(())
}
