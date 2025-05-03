mod data;
mod plot;

use anyhow::{anyhow, Result};
use rand::thread_rng;
use std::collections::HashMap;
use std::path::Path;
use tokio;

use data::{
    Admissions,
    AdmissionsOps, CsvLoad,
    AdmissionsOneHot, AdmissionsOneHotOps, AdmissionsOneHotPrint,
    RecordOH,
};

/// Create the output directory if it doesn't exist
async fn setup_output_directory(dir: &str) -> Result<()> {
    if !Path::new(dir).exists() {
        std::fs::create_dir_all(dir)
            .map_err(|e| anyhow!("cannot create '{}': {e}", dir))?;
    }
    Ok(())
}

/// Group scaled one‑hot rows by their **original** rank (r1–r4 back‑mapping)
fn group_by_rank(oh: &AdmissionsOneHot) -> HashMap<i32, Vec<RecordOH>> {
    let mut map = HashMap::new();
    for rec in &oh.rows {
        let rank = if rec.r1 == 1 {
            1
        } else if rec.r2 == 1 {
            2
        } else if rec.r3 == 1 {
            3
        } else if rec.r4 == 1 {
            4
        } else {
            0
        };
        map.entry(rank).or_insert_with(Vec::new).push(rec.clone());
    }
    map
}

/// Iterate over grouped data and call the new `plot_admissions_oh`
async fn plot_by_rank(grouped: HashMap<i32, Vec<RecordOH>>, out_dir: &str) -> Result<()> {
    for (rank, rows) in grouped {
        if rows.is_empty() {
            continue;
        }
        let png   = format!("{out_dir}/admissions_rank{rank}.png");
        let title = format!("Admissions – GRE vs GPA (Scaled) • Rank {rank}");
        plot::plot_admissions_oh(&rows, &png, &title)?;
        println!("wrote {png}");
    }
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    // ------------------------------------------------------------------ paths
    let csv  = "student-admissions/student_data.csv";
    let out  = "student-admissions/plots";
    let test = 0.10; // 10 %

    setup_output_directory(out).await?;

    // ------------------------------------------------------------- load → one‑hot
    let admissions        = Admissions::from_csv(csv)?;
    let mut admissions_oh = admissions.one_hot_rank();
    admissions_oh.scale_mut();

    // ------------------------------------------------------------- split
    let mut rng = thread_rng();
    let (train_oh, test_oh) = admissions_oh.split_train_test(test, &mut rng);

    // ------------------------------------------------------------- combine (for plotting only)
    let combined_oh: AdmissionsOneHot = train_oh
        .rows
        .into_iter()
        .chain(test_oh.rows.into_iter())
        .collect();

    // ------------------------------------------------------------- plot per rank
    let grouped = group_by_rank(&combined_oh);
    plot_by_rank(grouped, out).await?;

    Ok(())
}
