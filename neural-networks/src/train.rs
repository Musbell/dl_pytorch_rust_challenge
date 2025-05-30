use burn::prelude::*;
use burn::optim::{SgdConfig, Optimizer, GradientsParams};
use burn_ndarray::{NdArray, NdArrayDevice};
use std::error::Error;

use crate::model::{build_dataloaders, MnistTrainingConfig, NetworkConfig, Network};

type AutodiffNdArray = burn::backend::Autodiff<NdArray>;

fn create_artifact_dir(artifact_dir: &str) {
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn train(
    device: &NdArrayDevice,
    epochs: usize,
    learning_rate: f64,
) -> Result<(), Box<dyn Error>> {
    let config = MnistTrainingConfig::new();
    let (dataloader_train, _dataloader_test) = build_dataloaders(&config);

    // Build model
    let net_cfg = NetworkConfig::new();
    let mut model: Network<AutodiffNdArray> = net_cfg.init(device);
    
    // Initialize optimizer
    let mut optimizer = SgdConfig::new().init();

    println!("Number of train batches: {}", dataloader_train.iter().count());

    // Training loop
    for epoch in 0..epochs {
        let mut running_loss = 0.0;
        let mut batch_count = 0;

        for batch in dataloader_train.iter() {
            // Convert tensors to autodiff backend
            let images: Tensor<AutodiffNdArray, 3> = Tensor::from_data(batch.images.to_data(), device);
            let targets: Tensor<AutodiffNdArray, 1, Int> = Tensor::from_data(batch.targets.to_data(), device);
            
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

    Ok(())
}

/// Main training pipeline
pub fn run() -> Result<(), Box<dyn Error>> {
    create_artifact_dir("/tmp/burn-example-mnist");
    
    // Training parameters (matching PyTorch example)
    let epochs = 5;
    let learning_rate = 0.003;

    // Use autodiff backend for training
    let device = NdArrayDevice::default();

    // Train the model
    train(&device, epochs, learning_rate)?;

    Ok(())
}