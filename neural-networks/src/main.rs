// Declare the data module
mod data;
mod model;
mod train;

use train::run;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Run the training pipeline
    run()?;
    Ok(())
}
