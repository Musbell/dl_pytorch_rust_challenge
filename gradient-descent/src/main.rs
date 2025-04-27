mod data;
mod plot;
mod model;
mod train;
mod synthetic_data_generator;

use burn::prelude::Backend;
use anyhow::Result;
use burn::backend::{Autodiff, NdArray};
use plotters::prelude::*;
use std::path::PathBuf;
use burn::backend;
use crate::data::read_data;
use crate::synthetic_data_generator::generate_synthetic_csv;

#[tokio::main]
async fn main() -> Result<()> {
    // --- Constants ---
    const NUM_SAMPLES: usize = 1_000;
    const DATA_PATH: &str = "gradient-descent/data_demo.csv";
    const EPOCHS: usize = 500;
    const LR: f64 = 0.1;

    // --- Generate synthetic data ---
    let theta = [3.40318325, -3.18416003, -3.31863705];
    generate_synthetic_csv(NUM_SAMPLES, theta, DATA_PATH)
        .await
        .expect("Failed to generate synthetic data");
    println!("Wrote {} samples to {}", NUM_SAMPLES, DATA_PATH);

    // --- Load data ---
    let (admitted, rejected, feats, targs) = read_data(DATA_PATH)
        .await
        .expect("Failed to read data");
    println!("Loaded {} samples", feats.len());
    println!("Admitted: {} samples", admitted.len());
    println!("Rejected: {} samples", rejected.len());

    // --- Set up backend ---
    type MyNd = NdArray<f64>;
    type MyAuto = Autodiff<MyNd>;
    // let device = <MyNd as backend::Backend>::Device::default();
    let device = <MyNd as Backend>::Device::default();
    let graph_interval = Some((EPOCHS / 10).max(1));

    // --- Set up plot canvas ---
    let root = BitMapBackend::new("burn_logistic_plot.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let all_pts = admitted.iter().chain(&rejected);
    let (min_x, max_x, min_y, max_y) = all_pts.clone().fold(
        (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY),
        |(min_x, max_x, min_y, max_y), &(x, y)| {
            (min_x.min(x), max_x.max(x), min_y.min(y), max_y.max(y))
        }
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

    chart.configure_mesh()
        .x_desc("Feature 1")
        .y_desc("Feature 2")
        .draw()?;

    // --- Plot the admitted/rejected points ---
    plot::plot_points(&mut chart, &admitted, &rejected)?;

    // --- Train model and plot boundaries ---
    println!("Starting training…");
    let _model = train::train_and_plot::<MyAuto, _>(
        &mut chart,
        &x_range,
        feats,
        targs,
        EPOCHS,
        LR,
        graph_interval,
        &device,
    )?;

    // --- Draw legend and save the plot ---
    chart.configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .position(SeriesLabelPosition::UpperRight)
        .draw()?;

    root.present()?;
    println!("Saved plot to burn_logistic_plot.png");

    Ok(())
}
