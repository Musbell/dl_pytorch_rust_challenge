use crate::data::{FashionMNISTBatch, FashionMNISTBatcher};
use crate::dataset::FashionMNISTDataset;
use burn::config::Config;
use burn::data::dataloader::{DataLoader, DataLoaderBuilder};
use burn::module::Module;
use burn::backend::Wgpu;
use burn::nn::{Linear, LinearConfig};
use burn::prelude::{Backend, Int, Tensor};
use burn::tensor::activation::{log_softmax, relu};
use std::sync::Arc;

// We're using Wgpu as our backend

/// Alias for a data loader over Wgpu batches
pub type WgpuDataLoader = Arc<dyn DataLoader<Wgpu, FashionMNISTBatch<Wgpu>>>;



#[derive(Config)]
pub struct MnistTrainingConfig {
    #[config(default = 10)]
    num_epochs: usize,

    #[config(default = 64)]
    batch_size: usize,

    #[config(default = 4)]
    num_workers: usize,

    #[config(default = 42)]
    seed: u64,
}

pub fn build_dataloaders(config: &MnistTrainingConfig) -> (WgpuDataLoader, WgpuDataLoader) {
    let batcher = FashionMNISTBatcher::default();
    let train = DataLoaderBuilder::<Wgpu, _, _>::new(batcher.clone())
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(FashionMNISTDataset::train());
    let test = DataLoaderBuilder::<Wgpu, _, _>::new(batcher)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(FashionMNISTDataset::test());
    (train, test)
}



// === Define a multi-layer Network module ===
#[derive(Config, Debug)]
pub struct NetworkConfig {
    #[config(default = 784)]
    input_size: usize,
    #[config(default = 128)]
    hidden1: usize,
    #[config(default = 64)]
    hidden2: usize,
    #[config(default = 10)]
    output_size: usize,
}

#[derive(Module, Debug)]
pub struct Network<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    fc3: Linear<B>,
}

impl NetworkConfig {
    pub(crate) fn init<B: Backend>(&self, device: &B::Device) -> Network<B> {
        Network {
            fc1: LinearConfig::new(self.input_size, self.hidden1).init::<B>(device),
            fc2: LinearConfig::new(self.hidden1, self.hidden2).init::<B>(device),
            fc3: LinearConfig::new(self.hidden2, self.output_size).init::<B>(device),
        }
    }
}

impl<B: Backend> Network<B> {
    pub(crate) fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x1 = relu(self.fc1.forward(input));
        let x2 = relu(self.fc2.forward(x1));
        let logits = self.fc3.forward(x2);
        let output = log_softmax(logits, 1);
        output
    }

    /// Compute negative log likelihood loss from log probabilities and integer targets.
    pub fn loss(
        &self,
        log_probs: Tensor<B, 2>,
        targets: Tensor<B, 1, Int>,
        _device: &B::Device,
    ) -> Tensor<B, 1> {
        // Since we already have log probabilities, we compute NLL manually
        // NLL loss = -log_probs[target_class] for each sample
        let num_classes = log_probs.dims()[1];

        // Convert targets to one-hot then compute NLL
        let targets_one_hot = targets.one_hot(num_classes).float();
        let loss_per_sample = -(log_probs * targets_one_hot).sum_dim(1);
        loss_per_sample.mean()
    }
}
