use anyhow::{anyhow, Result};
use burn::prelude::*;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::{Tensor, TensorData, Element};
use burn::tensor::activation::sigmoid;
use burn::tensor::backend::AutodiffBackend;

/// A simple 2-D lostic regression.
#[derive(Module, Debug)]
pub struct LogisticRegression<B: Backend> {
    linear: Linear<B>,
}

impl <B: Backend> LogisticRegression<B> {
    /// Create a new logistics regression model.
    pub fn new(input_dim: usize, device: &B::Device) -> Self {
        let config = LinearConfig::new(input_dim, 1).with_bias(true);
        let linear = config.init(device);

        Self { linear }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        sigmoid(self.linear.forward(input))
    }

    /// Extract weights + bias into CPU `Vec<f64>`
    pub fn get_params(&self) -> Result<(Vec<f64>, f64)> {
        let cpu = Default::default();
        let w = self.linear.weight.val().to_device(&cpu).to_data()
        .into_vec().map_err(|e| anyhow!("Bias error: {:?}", e))?;
        let b = self.linear.bias.as_ref().unwrap().val().to_device(&cpu).to_data()
        .into_vec().map_err(|e| anyhow!("Bias error: {:?}", e))?;

        Ok((w, b[0]))
    }
}

/// Binary cross‑entropy loss.
pub fn bce_loss<B>(
    pred: Tensor<B, 2>,
    tgt: Tensor<B, 2>,
) -> Tensor<B, 1>
where
    B: AutodiffBackend<FloatElem = f64>,
    B::FloatElem: Element + std::ops::Sub<Output = B::FloatElem> + PartialOrd + Copy + std::fmt::Debug + 'static,
{
    let eps = B::FloatElem::from_elem(1e-10);
    let one = B::FloatElem::from_elem(1.0);
    let p = pred.clamp(eps, one - eps);
    let t1 = tgt.clone() * p.clone().log();
    let t2 = (tgt.ones_like() - tgt) * (p.ones_like() - p).log();
    (t1 + t2).neg().mean()
}

/// Compute accuracy by thresholding at 0.5.
pub fn accuracy<B: Backend<FloatElem = f64>>(pred: Tensor<B, 2>, tgt: Tensor<B, 2>) -> f64 {
    pred.greater_equal_elem(0.5)
        .equal(tgt.greater_equal_elem(0.5))
        .float()
        .mean()
        .into_scalar()
}