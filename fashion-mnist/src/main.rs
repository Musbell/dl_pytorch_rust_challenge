mod data;
mod model;
mod train;
mod dataset;
mod visualization;

use train::run;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Run the training pipeline
    run()?;
    Ok(())
}
