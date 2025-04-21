use anyhow::{anyhow, Result};

use burn::prelude::*;                           // Module derive, Tensor, etc.
use burn::backend::{Autodiff, NdArray};
use burn::nn::{Linear, LinearConfig};
use burn::optim::{SgdConfig, Optimizer, GradientsParams};
use burn::tensor::{Tensor, TensorData, Element};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::activation::sigmoid;

use csv::ReaderBuilder;
use plotters::prelude::*;
use plotters::coord::types::RangedCoordf64;
use serde::Deserialize;
use std::path::PathBuf;
use std::error::Error;

// --- CSV record struct ---
#[derive(Debug, Deserialize)]
struct Record {
    #[serde(rename = "0")] x: f64,
    #[serde(rename = "1")] y: f64,
    #[serde(rename = "2")] label: i32,
}

/// Load points + features + targets from CSV
fn read_data(path: &str) -> Result<(
    Vec<(f64, f64)>, // admitted points
    Vec<(f64, f64)>, // rejected points
    Vec<Vec<f64>>,   // features for training
    Vec<f64>         // labels for training
)> {
    let mut rdr = ReaderBuilder::new().has_headers(false).from_path(path)?;
    let mut admitted = Vec::new();
    let mut rejected = Vec::new();
    let mut feats = Vec::new();
    let mut targs = Vec::new();

    for rec in rdr.deserialize::<Record>() {
        let r = rec?;
        feats.push(vec![r.x, r.y]);
        if r.label == 1 {
            admitted.push((r.x, r.y));
            targs.push(1.0);
        } else {
            rejected.push((r.x, r.y));
            targs.push(0.0);
        }
    }

    if feats.is_empty() {
        return Err(anyhow!("No data loaded from {}", path));
    }
    Ok((admitted, rejected, feats, targs))
}

/// Plot admitted/rejected points.
fn plot_points<DB: DrawingBackend>(
    chart: &mut ChartContext<DB, Cartesian2d<RangedCoordf64, RangedCoordf64>>,
    admitted: &[(f64, f64)],
    rejected: &[(f64, f64)],
) -> Result<(), DrawingAreaErrorKind<DB::ErrorType>> {
    chart.draw_series(
        rejected.iter().map(|&(x, y)| Circle::new((x, y), 3, BLUE.filled()))
    )?
        .label("Rejected (0)")
        .legend(|(x, y)| Circle::new((x, y), 5, BLUE.filled()));

    chart.draw_series(
        admitted.iter().map(|&(x, y)| Circle::new((x, y), 3, RED.filled()))
    )?
        .label("Admitted (1)")
        .legend(|(x, y)| Circle::new((x, y), 5, RED.filled()));

    Ok(())
}

/// Draw a linear decision boundary: w0*x + w1*y + b = 0.
fn draw_boundary_line<'a, DB: DrawingBackend>(
    chart: &mut ChartContext<'a, DB, Cartesian2d<RangedCoordf64, RangedCoordf64>>,
    x_range: &std::ops::Range<f64>,
    weights: &[f64],
    bias: f64,
    color: PaletteColor<Palette99>,
    label: &str,
) -> Result<(), DrawingAreaErrorKind<DB::ErrorType>> {
    if weights.len() < 2 || weights[1].abs() < 1e-8 {
        return Ok(());
    }
    let (w0, w1) = (weights[0], weights[1]);
    let y0 = (-w0 * x_range.start - bias) / w1;
    let y1 = (-w0 * x_range.end   - bias) / w1;

    chart.draw_series(LineSeries::new(
        vec![(x_range.start, y0), (x_range.end, y1)],
        color.stroke_width(2),
    ))?
        .label(label)
        .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color.stroke_width(2)));

    Ok(())
}

#[derive(Module, Debug)]
pub struct LogisticRegression<B: Backend> {
    linear: Linear<B>,
}

impl<B: Backend> LogisticRegression<B> {
    pub fn new(input_dim: usize, device: &B::Device) -> Self {
        // Use default Xavier initializer
        let config = LinearConfig::new(input_dim, 1)
            .with_bias(true);
        let linear = config.init(device);
        Self { linear }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        sigmoid(self.linear.forward(input))
    }

    pub fn get_params(&self) -> Result<(Vec<f64>, f64)> {
        let cpu = Default::default();
        let w_data = self.linear.weight.val().to_device(&cpu).to_data();
        let b_data = self.linear.bias.as_ref().unwrap().val().to_device(&cpu).to_data();
        let w_vec = w_data.into_vec().map_err(|e| anyhow!("Weight extract error: {:?}", e))?;
        let b_vec = b_data.into_vec().map_err(|e| anyhow!("Bias extract error: {:?}", e))?;
        Ok((w_vec, b_vec[0]))
    }
}

/// Binary cross‑entropy loss.
fn bce_loss<B>(
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
fn accuracy<B: Backend<FloatElem = f64>>(pred: Tensor<B, 2>, tgt: Tensor<B, 2>) -> f64 {
    pred.greater_equal_elem(0.5)
        .equal(tgt.greater_equal_elem(0.5))
        .float()
        .mean()
        .into_scalar()
}

/// Train for `epochs`, log and plot boundaries.
fn train_and_plot<B, DB>(
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
    DB::ErrorType: Error + Send + Sync + 'static,
{
    let n = feats.len();
    let d = feats[0].len();
    let flat = feats.into_iter().flatten().collect::<Vec<_>>();
    let x = Tensor::<B, 2>::from_floats(TensorData::new(flat, [n, d]), device);
    let y = Tensor::<B, 2>::from_floats(TensorData::new(targs, [n, 1]), device);

    let mut model = LogisticRegression::new(d, device);
    let mut opt = SgdConfig::new().init();
    let print_int = (epochs / 10).max(1);
    let mut last_loss: Option<f64> = None;

    for epoch in 0..epochs {
        let pred = model.forward(x.clone());
        let loss_t = bce_loss(pred.clone(), y.clone());
        let loss_val = loss_t.clone().into_scalar();

        let grads = GradientsParams::from_grads(loss_t.backward(), &model);
        model = opt.step(lr, model, grads);

        if epoch % print_int == 0 || epoch == epochs - 1 {
            println!("\n========== Epoch {:>3} ==========", epoch);
            if let Some(prev) = last_loss {
                if loss_val > prev {
                    println!("Train loss: {:.6}  WARNING - Loss Increasing", loss_val);
                } else {
                    println!("Train loss: {:.6}", loss_val);
                }
            } else {
                println!("Train loss: {:.6}", loss_val);
            }
            last_loss = Some(loss_val);

            let acc = accuracy(model.clone().no_grad().forward(x.clone()), y.clone());
            println!("Accuracy: {:.4}", acc);
        }

        if let Some(interval) = graph_every {
            if epoch % interval == 0 || epoch == epochs - 1 {
                let (w, b) = model.get_params()?;
                let color = Palette99::pick(epoch % 99);
                draw_boundary_line(chart, x_range, &w, b, color, &format!("E{}", epoch))?;
            }
        }
    }
    Ok(model)
}

fn main() -> Result<()> {
    type MyNd   = NdArray<f64>;
    type MyAuto = Autodiff<MyNd>;
    let device = <MyNd as Backend>::Device::default();

    let epochs = 200;
    let lr     = 0.1;
    // 10% plotting interval (every 10% of epochs)
    let graph_interval = Some((epochs / 10).max(1));

    let data_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("data.csv")
        .to_string_lossy()
        .into_owned();
    let (adm, rej, feats, targs) = read_data(&data_path)?;
    println!("Loaded {} points ({} admitted, {} rejected)", feats.len(), adm.len(), rej.len());

    let root = BitMapBackend::new("burn_logistic_plot.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;
    let all_pts = adm.iter().chain(&rej);
    let (min_x, max_x, min_y, max_y) = all_pts.clone().fold(
        (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY),
        |(min_x, max_x, min_y, max_y), &(x, y)| (min_x.min(x), max_x.max(x), min_y.min(y), max_y.max(y)),
    );
    let pad_x = (max_x - min_x).max(0.1) * 0.1;
    let pad_y = (max_y - min_y).max(0.1) * 0.1;
    let x_range = (min_x - pad_x)..(max_x + pad_x);
    let y_range = (min_y - pad_y)..(max_y + pad_y);

    let mut chart = ChartBuilder::on(&root)
        .caption("Logistic Regression with Burn", ("sans-serif", 30))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(x_range.clone(), y_range.clone())?;

    chart.configure_mesh().x_desc("Feature 1").y_desc("Feature 2").draw()?;
    plot_points(&mut chart, &adm, &rej)?;

    println!("Starting training…");
    let _ = train_and_plot::<MyAuto, _>(
        &mut chart,
        &x_range,
        feats,
        targs,
        epochs,
        lr,
        graph_interval,
        &device,
    )?;

    chart.configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .position(SeriesLabelPosition::UpperRight)
        .draw()?;
    root.present()?;
    println!("Saved plot to burn_logistic_plot.png");
    Ok(())
}
