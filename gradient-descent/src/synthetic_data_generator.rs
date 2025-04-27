use rand::Rng;
use csv::Writer;
use std::error::Error; 

/// Generate and save `n` synthetic data points to a CSV file at `path`.
/// - `n`: number of samples to generate (can be in the millions).
/// - `theta`: logistic parameters [bias, weight_x1, weight_x2].
///
/// Points (x1, x2) are drawn uniformly from [0, 1].
/// A label of `1` is assigned whenever `theta[0] + theta[1]*x1 + theta[2]*x2 >= 0.0`, else `0`.
pub async fn generate_synthetic_csv(
    n: usize,
    theta: [f64; 3],
    path: &str,
) -> Result<(), Box<dyn Error>> { // Consider using anyhow::Result<()> here too
    let mut rng = rand::thread_rng();
    let mut wtr = Writer::from_path(path)?;
    // Write CSV header - these are the names the reader expects
    wtr.write_record(&["x", "y", "label"])?;

    // Generate and write each record
    for _ in 0..n {
        let x1 = rng.gen::<f64>();
        let x2 = rng.gen::<f64>();
        let z = theta[0] + theta[1] * x1 + theta[2] * x2;
        let label = if z >= 0.0 { 1 } else { 0 };
        wtr.write_record(&[
            x1.to_string(),
            x2.to_string(),
            label.to_string(),
        ])?;
    }

    wtr.flush()?;
    Ok(())
}
