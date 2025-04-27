use anyhow::Result;
use burn::optim::{SgdConfig, Optimizer, GradientsParams};
use burn::prelude::{Module, Tensor, TensorData};
use burn::tensor::backend::AutodiffBackend;
use plotters::prelude::*;
use plotters::coord::types::RangedCoordf64;

use crate::model::{LogisticRegression, bce_loss, accuracy};

/// Train for `epochs`, log progress, and optionally draw boundaries.
pub fn train_and_plot<B, DB>(
    chart: &mut ChartContext<DB, Cartesian2d<RangedCoordf64, RangedCoordf64>>,
    x_range: &std::ops::Range<f64>,
    feats: Vec<Vec<f64>>,
    targs: Vec<f64>,
    epochs: usize,
    lr: f64,
    graph_every: Option<usize>,
    device: &B::Device,
) -> Result<LogisticRegression<B>, anyhow::Error>
where
    B: AutodiffBackend<FloatElem = f64>,
    DB: DrawingBackend,
    DB::ErrorType: std::error::Error + Send + Sync + 'static,
{
    // Prepare data tensors
    let n = feats.len();
    let d = feats[0].len();
    let flat = feats.into_iter().flatten().collect::<Vec<_>>();
    let x = Tensor::<B, 2>::from_floats(TensorData::new(flat, [n, d]), device);
    let y = Tensor::<B, 2>::from_floats(TensorData::new(targs, [n, 1]), device);

    // Model & optimizer
    let mut model = LogisticRegression::new(d, device);
    let mut opt = SgdConfig::new().init();
    let print_int = (epochs / 10).max(1);
    let mut last_loss: Option<f64> = None;

    for epoch in 0..epochs {
        let pred = model.forward(x.clone());
        let loss_t = bce_loss(pred.clone(), y.clone());
        let loss_val = loss_t.clone().into_scalar();

        // Backward & step
        let grads = GradientsParams::from_grads(loss_t.backward(), &model);
        model = opt.step(lr, model, grads);

        // Logging
        if epoch % print_int == 0 || epoch == epochs - 1 {
            println!("\n========== Epoch {:>3} ==========", epoch);
            if let Some(prev) = last_loss {
                let warning = if loss_val > prev { " WARNING - Loss ↑" } else { "" };
                println!("Train loss: {:.6}{}", loss_val, warning);
            } else {
                println!("Train loss: {:.6}", loss_val);
            }
            last_loss = Some(loss_val);

            let acc = accuracy(model.clone().no_grad().forward(x.clone()), y.clone());
            println!("Accuracy: {:.4}", acc);
        }

        // Draw boundary if requested
        if let Some(interval) = graph_every {
            if epoch % interval == 0 || epoch == epochs - 1 {
                let (w, b) = model.get_params()?;
                let color = Palette99::pick(epoch % 99);
                crate::plot::draw_boundary_line(chart, x_range, &w, b, color, &format!("E{}", epoch))?;
            }
        }
    }

    Ok(model)
}
