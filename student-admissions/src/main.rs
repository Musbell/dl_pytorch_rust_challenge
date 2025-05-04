// ─────────────────────────── main.rs ────────────────────────────────────────
mod data;
mod plot;
mod train;
mod model;

use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::path::Path;
use tokio;
use rand::{SeedableRng, rngs::StdRng};

use data::{
    Admissions, AdmissionsOps, CsvLoad,
    AdmissionsOneHot, AdmissionsOneHotOps, RecordOH, XY,
};
use model::{BCE, MSE}; // Swap BCE for MSE or vice‑versa in train_nn::<…>
use train::train_nn;
use crate::data::AdmissionsOneHotPrint;
/* fs helper */
async fn setup_output_directory(dir: &str) -> Result<()> {
    if !Path::new(dir).exists() {
        std::fs::create_dir_all(dir)
            .map_err(|e| anyhow!("cannot create '{dir}': {e}"))?;
    }
    Ok(())
}

/* back‑map r1..r4 → rank */
fn group_by_rank(oh: &AdmissionsOneHot) -> HashMap<i32, Vec<RecordOH>> {
    let mut map: HashMap<i32, Vec<RecordOH>> = HashMap::new();
    for rec in &oh.rows {
        let rank = match (rec.r1, rec.r2, rec.r3, rec.r4) {
            (1, _, _, _) => 1,
            (_, 1, _, _) => 2,
            (_, _, 1, _) => 3,
            (_, _, _, 1) => 4,
            _            => 0,
        };
        map.entry(rank).or_default().push(rec.clone());
    }
    map
}

/* plotting helper */
async fn plot_by_rank(grouped: HashMap<i32, Vec<RecordOH>>, out: &str) -> Result<()> {
    for (rank, rows) in grouped {
        if rows.is_empty() { continue; }
        let png   = format!("{out}/admissions_rank{rank}.png");
        let title = format!("Admissions • GRE vs GPA (scaled) • Rank {rank}");
        plot::plot_admissions_oh(&rows, &png, &title)?;
    }
    Ok(())
}

/* tiny sigmoid */
#[inline]
fn sigmoid(v: f64) -> f64 { 1.0 / (1.0 + (-v).exp()) }

/* main */
#[tokio::main]
async fn main() -> Result<()> {
    /* config */
    let csv    = "student-admissions/student_data.csv";
    let out    = "student-admissions/plots";
    let test   = 0.25;
    let epochs = 10_000;
    let lr     = 5e-4;

    setup_output_directory(out).await?;

    /* load CSV */
    let admissions = Admissions::from_csv(csv)?;

    /* preview raw CSV */
    println!("\n=== RAW (first 5 rows) ===");
    admissions.head(5);

    /* one‑hot → scale */
    let mut admissions_oh = admissions.one_hot_rank();
    admissions_oh.scale_mut();

    /* preview processed data */
    println!("\n=== ONE‑HOT + SCALED (first 5 rows) ===");
    admissions_oh.head_oh(5);

    /* split train / test */
    let mut rng = StdRng::seed_from_u64(42);
    let (train_oh, test_oh) = admissions_oh.split_train_test(test, &mut rng);

    let XY { x: x_train, y: y_train } = train_oh.split_xy();
    let XY { x: x_test,  y: y_test  } = test_oh .split_xy();

    /* train */
    let weights = train_nn::<BCE>(&XY { x: x_train, y: y_train }, epochs, lr)?;

    /* evaluate */
    let mut correct = 0usize;
    for (x, &y) in x_test.iter().zip(&y_test) {
        let z   = x.iter().zip(&weights).map(|(xi, wi)| xi * wi).sum::<f64>();
        if (sigmoid(z) > 0.5) == (y == 1) { correct += 1; }
    }
    println!("prediction accuracy: {:.3}",
             correct as f64 / y_test.len() as f64);

    /* plots */
    let combined_oh: AdmissionsOneHot = train_oh
        .rows.into_iter()
        .chain(test_oh.rows.into_iter())
        .collect();

    plot_by_rank(group_by_rank(&combined_oh), out).await?;
    Ok(())
}
