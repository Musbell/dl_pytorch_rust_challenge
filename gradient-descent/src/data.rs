use anyhow::{anyhow, Result};
use csv::ReaderBuilder;
use serde::Deserialize;

// --- CSV record struct ---
// The original struct used #[serde(rename = "0")] etc.
// This tells serde to look for headers named "0", "1", "2".
// However, the generate_synthetic_csv function writes headers "x", "y", "label".
// We need the struct field names (or their renamed versions) to match the actual header names.
// Since the struct field names (x, y, label) match the desired header names,
// we can simply remove the #[serde(rename)] attributes. Serde will default to
// matching field names to header names.
#[derive(Debug, Deserialize)]
struct Record {
    x: f64,
    y: f64,
    label: i32,
}

/// Load points + features + targets from CSV
pub async fn read_data(path: &str) -> Result<(
    Vec<(f64, f64)>, // admitted points
    Vec<(f64, f64)>, // rejected points
    Vec<Vec<f64>>,   // features for training
    Vec<f64>         // labels for training
)> {
    // We expect headers "x", "y", "label" based on how generate_synthetic_csv writes them.
    // The `Record` struct without `#[serde(rename)]` will map to these.
    let mut rdr = ReaderBuilder::new().has_headers(true).from_path(path)?;
    let mut admitted = Vec::new();
    let mut rejected = Vec::new();
    let mut feats = Vec::new();
    let mut targs = Vec::new();

    // Deserialize records, mapping CSV columns "x", "y", "label" to struct fields.
    for rec in rdr.deserialize::<Record>() {
        let r = rec?; // Propagate any deserialization errors
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