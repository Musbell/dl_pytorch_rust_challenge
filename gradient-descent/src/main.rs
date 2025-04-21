use plotters::prelude::*;
use plotters::coord::types::RangedCoordf64;
use csv::ReaderBuilder;
use serde::Deserialize;
use anyhow::Result; // Use anyhow for convenient error handling
use std::path::PathBuf;
use std::env; // To use env! macro

// --- Data Structure ---
#[derive(Debug, Deserialize)]
struct Record {
    #[serde(rename = "0")]
    x: f64,
    #[serde(rename = "1")]
    y: f64,
    #[serde(rename = "2")]
    label: i32,
}

// --- Data Reading Function ---
fn read_and_separate_data(path: &str) -> Result<(Vec<(f64, f64)>, Vec<(f64, f64)>)> {
    let mut rdr = ReaderBuilder::new()
        .has_headers(false)
        .from_path(path)?;
    let mut admitted = Vec::new();
    let mut rejected = Vec::new();
    for result in rdr.deserialize::<Record>() {
        let record = result?;
        if record.label == 1 {
            admitted.push((record.x, record.y));
        } else {
            rejected.push((record.x, record.y));
        }
    }
    if admitted.is_empty() && rejected.is_empty() {
        // Use eprintln! for warnings/errors to send to stderr
        eprintln!("Warning: No data points loaded from {}", path);
    }
    Ok((admitted, rejected))
}

// --- Plotting Functions ---

/// Plots rejected (blue) and admitted (red) points onto the chart.
fn plot_points<DB: DrawingBackend>(
    chart: &mut ChartContext<DB, Cartesian2d<RangedCoordf64, RangedCoordf64>>,
    admitted: &[(f64, f64)],
    rejected: &[(f64, f64)],
) -> Result<(), DrawingAreaErrorKind<DB::ErrorType>> {
    chart
        .draw_series(
            rejected
                .iter()
                .map(|&(x, y)| Circle::new((x, y), 5, BLUE.filled())),
        )?
        .label("Rejected (0)")
        .legend(|(x, y)| Circle::new((x, y), 5, BLUE.filled()));

    chart
        .draw_series(
            admitted
                .iter()
                .map(|&(x, y)| Circle::new((x, y), 5, RED.filled())),
        )?
        .label("Admitted (1)")
        .legend(|(x, y)| Circle::new((x, y), 5, RED.filled()));

    Ok(())
}


// --- Main Function ---
fn main() -> Result<()> {
    // Construct the path to data.csv reliably.
    // Assumes data.csv is in the same directory as Cargo.toml
    let data_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("data.csv") // Make sure 'data.csv' exists here!
        .to_string_lossy()
        .into_owned();

    println!("Attempting to read data from: {}", data_path);

    // Read and separate data points.
    let (admitted_points, rejected_points) = read_and_separate_data(&data_path)?;

    // Check if data was actually loaded
    if admitted_points.is_empty() && rejected_points.is_empty() {
        eprintln!("Error: No data points were successfully read. Please ensure '{}' exists and contains valid data.", data_path);
        // Optionally, return an error or exit early
        // return Err(anyhow::anyhow!("Failed to load data points"));
        // Or:
        // std::process::exit(1);
    } else {
        println!("Loaded {} admitted and {} rejected points.", admitted_points.len(), rejected_points.len());
    }


    // Setup the drawing area and chart.
    let output_path = "initial_plot.png"; // Changed output name slightly
    let img_width = 800;
    let img_height = 600;
    let root = BitMapBackend::new(output_path, (img_width, img_height)).into_drawing_area();
    root.fill(&WHITE)?;

    // --- Determine appropriate axis ranges (Optional but good practice) ---
    // Find min/max x and y from *both* datasets combined
    let all_points = admitted_points.iter().chain(rejected_points.iter());
    let (min_x, max_x, min_y, max_y) = all_points.fold(
        (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY),
        |(min_x, max_x, min_y, max_y), &(x, y)| {
            (min_x.min(x), max_x.max(x), min_y.min(y), max_y.max(y))
        },
    );

    // Add some padding to the ranges
    let padding_x = (max_x - min_x) * 0.05;
    let padding_y = (max_y - min_y) * 0.05;

    // Handle case where all points are identical or no points exist
    let x_coord_range = if min_x.is_finite() && max_x.is_finite() {
        (min_x - padding_x)..(max_x + padding_x)
    } else {
        -0.05..1.05 // Default range if no data
    };
    let y_coord_range = if min_y.is_finite() && max_y.is_finite() {
        (min_y - padding_y)..(max_y + padding_y)
    } else {
        -0.05..1.05 // Default range if no data
    };


    // --- Build Chart ---
    let mut chart = ChartBuilder::on(&root)
        .caption("Student Admission Data", ("sans-serif", 30).into_font())
        .margin(20) // Increased margin slightly for labels
        .x_label_area_size(50) // Increased size for labels
        .y_label_area_size(50) // Increased size for labels
        .build_cartesian_2d(x_coord_range.clone(), y_coord_range.clone())?; // Use calculated ranges

    chart
        .configure_mesh()
        .x_desc("Exam Score 1")
        .y_desc("Exam Score 2")
        .axis_desc_style(("sans-serif", 15)) // Style axis descriptions
        .draw()?;

    // Plot the data points.
    plot_points(&mut chart, &admitted_points, &rejected_points)?;
    

    // Configure and draw legend.
    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .position(SeriesLabelPosition::UpperRight) // Position legend
        .draw()?;

    // Save the plot to file.
    root.present()?; // Ensure drawing is finalized
    println!("Plot saved to {}", output_path);

    Ok(())
}