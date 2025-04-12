// --- LIBRARIES ---
use ndarray::prelude::*;
use ndarray::{Array, Array1, Array2}; // Removed Axis, s as they were unused
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use rand::SeedableRng;
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use std::path::PathBuf;

use std::{
    error::Error,
    fs::File,
    process,
    path::Path,
};

// Add plotters imports
use plotters::prelude::*;

// --- CSV Reader ---
// (Keep the csv_reader function as it was in the previous corrected version)
fn csv_reader<P: AsRef<Path>>(file_path: P) -> Result<(Array2<f64>, Array1<i32>), Box<dyn Error>> {
    let file = File::open(file_path)?;
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false) // Our data doesn't have headers
        .from_reader(file);

    let mut features: Vec<f64> = Vec::new();
    let mut labels: Vec<i32> = Vec::new();
    let mut num_features = 0;

    for (i, result) in rdr.records().enumerate() {
        let record = result?;
        if i == 0 {
            num_features = record.len() - 1; // Last column is the label
            if num_features == 0 {
                return Err("CSV rows have no features (only one column found).".into());
            }
        } else if record.len() - 1 != num_features {
            // Basic validation
            return Err(format!("Row {} has {} features, expected {}", i+1, record.len() - 1, num_features).into());
        }

        // Parse features (all except the last column)
        for feature_str in record.iter().take(num_features) {
            features.push(feature_str.trim().parse::<f64>()?);
        }
        // Parse label (last column)
        labels.push(record.get(num_features).ok_or("Missing label column")?.trim().parse::<i32>()?);
    }

    let num_samples = labels.len();
    if num_samples == 0 {
        return Err("CSV file is empty or contains no data rows.".into());
    }

    // Convert Vecs to ndarray Arrays
    let inputs = Array::from_shape_vec((num_samples, num_features), features)?;
    let target = Array::from_vec(labels);

    Ok((inputs, target))
}


// --- Step Function ---
// (Keep the step_function as it was)
fn step_function(t: f64) -> f64 {
    if t >= 0.0 { 1.0 } else { 0.0 }
}

// --- Prediction ---
// (Keep the prediction function as it was)
fn prediction(weights: &Array1<f64>, inputs: &ArrayView1<f64>, bias: f64) -> f64 {
    step_function(inputs.dot(weights) + bias)
}

// --- Perceptron Step ---
// (Keep the perceptron_step function as it was, perhaps use float tolerance)
fn perceptron_step(
    weights: &mut Array1<f64>,
    bias: &mut f64,
    inputs: &Array2<f64>,
    target: &Array1<i32>,
    learning_rate: f64,
) {
    let num_samples = inputs.nrows();

    for i in 0..num_samples {
        let input_row = inputs.row(i);
        let prediction_value = prediction(weights, &input_row, *bias);
        let target_val = target[i] as f64;

        let error = target_val - prediction_value;

        // Use tolerance for float comparison
        if error.abs() > 1e-9 {
            let update_factor = error * learning_rate;
            weights.scaled_add(update_factor, &input_row);
            *bias += update_factor;
        }
    }
}

// --- Training Algorithm ---
// (Keep the train_perceptron_algorithm function as it was)
fn train_perceptron_algorithm(
    inputs: &Array2<f64>,
    target: &Array1<i32>,
    learning_rate: f64,
    epochs: usize,
    rng: &mut ChaCha8Rng,
) -> Vec<(f64, f64)> {

    let num_samples = inputs.nrows();
    let num_features = inputs.ncols();

    if num_samples == 0 || num_features == 0 {
        panic!("Input data is empty.");
    }
    if num_features != 2 {
        println!(
            "Warning: Boundary line calculation/plotting assumes 2 features, but found {}.",
            num_features
        );
    }

    // --- Initialization ---
    let mut first_col_max = inputs
        .column(0)
        .iter()
        .fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));

    if !first_col_max.is_finite() {
        println!("Warning: Could not determine valid maximum of the first feature column. Using 0.0 for bias initialization offset.");
        first_col_max = 0.0;
    }

    let mut w: Array1<f64> = Array::random_using(num_features, Uniform::new(0.0, 1.0), rng);
    let mut b: f64 = rng.gen_range(0.0..1.0) + first_col_max;

    // --- Training Loop ---
    let mut boundary_lines: Vec<(f64, f64)> = Vec::with_capacity(epochs);

    println!("Initial W: {:.4?}", w);
    println!("Initial b: {:.4}", b);

    // Use _epoch to mark the variable as intentionally unused unless printing progress
    for _epoch in 0..epochs {
        perceptron_step(&mut w, &mut b, inputs, target, learning_rate);

        if num_features == 2 {
            if w.len() == 2 {
                if w[1].abs() > 1e-10 {
                    let slope = -w[0] / w[1];
                    let intercept = -b / w[1];
                    boundary_lines.push((slope, intercept));
                } else {
                    boundary_lines.push((f64::NAN, f64::NAN));
                }
            } else {
                boundary_lines.push((f64::NAN, f64::NAN)); // Should not happen
            }
        }
        // Optional: Print progress
        // if _epoch % 5 == 0 || _epoch == epochs - 1 {
        //     println!("Epoch: {}, W: {:.4?}, b: {:.4}", _epoch, w, b);
        // }
    }

    println!("Training finished.");
    println!("Final W: {:.4?}", w);
    println!("Final b: {:.4}", b);
    boundary_lines
}


// --- Plotting Function ---
fn plot_perceptron(
    output_path: &str,
    title: &str,
    inputs: &Array2<f64>,
    target: &Array1<i32>,
    boundary_lines: &Vec<(f64, f64)>,
) -> Result<(), Box<dyn Error>> {
    // Ensure we have 2 features for plotting
    if inputs.ncols() != 2 {
        return Err("Plotting requires exactly 2 features.".into());
    }

    // Setup drawing area
    let root_area = BitMapBackend::new(output_path, (800, 600)).into_drawing_area();
    root_area.fill(&WHITE)?;

    // Determine plot bounds with padding
    let (mut x_min, mut x_max) = inputs.column(0).iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(min_acc, max_acc), &x| (min_acc.min(x), max_acc.max(x)));
    let (mut y_min, mut y_max) = inputs.column(1).iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(min_acc, max_acc), &y| (min_acc.min(y), max_acc.max(y)));

    // Add padding
    let x_padding = (x_max - x_min) * 0.1;
    let y_padding = (y_max - y_min) * 0.1;
    x_min -= x_padding;
    x_max += x_padding;
    y_min -= y_padding;
    y_max += y_padding;

    // Create chart context
    let mut chart = ChartBuilder::on(&root_area)
        .caption(title, ("sans-serif", 30).into_font())
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)?;

    // Configure mesh
    chart.configure_mesh()
        .x_desc("Feature 1")
        .y_desc("Feature 2")
        .axis_desc_style(("sans-serif", 15))
        .draw()?;

    // --- Draw Scatter Plot ---
    chart.draw_series(
        inputs.outer_iter().zip(target.iter()).map(|(row, &label)| {
            let x = row[0];
            let y = row[1];
            let color = if label == 1 { &RED } else { &BLUE };
            Circle::new((x, y), 3, color.filled()) // Small filled circles
        })
    )?;


    // --- Draw Boundary Lines ---
    let num_lines = boundary_lines.len();
    for (idx, &(slope, intercept)) in boundary_lines.iter().enumerate() {
        if slope.is_nan() || intercept.is_nan() {
            continue; // Skip vertical lines or errors
        }

        // Calculate line endpoints based on x-axis range
        let y1 = slope * x_min + intercept;
        let y2 = slope * x_max + intercept;

        // Determine color/style (e.g., make final line thicker/darker)
        let mut line_color = RGBAColor(0, 0, 0, 0.15); // Default: faint black
        let mut stroke_width = 1;

        if idx == num_lines - 1 { // Highlight the final line
            line_color = BLACK.to_rgba(); // Solid black
            stroke_width = 2;
        } else if idx == 0 { // Slightly highlight the initial line
            line_color = RGBAColor(128, 128, 128, 0.5); // Greyish
            stroke_width = 1;
        }

        chart.draw_series(LineSeries::new(
            vec![(x_min, y1), (x_max, y2)],
            ShapeStyle {
                color: line_color,
                filled: false,
                stroke_width,
            },
        ))?;
    }

    // Add a legend (optional, slightly more complex for dynamic lines)
    // We can at least add a legend for the points
    chart.configure_series_labels()
        .border_style(&BLACK)
        .background_style(&WHITE.mix(0.8))
        .label_font(("sans-serif", 12))
        .draw()?;
    // Need to manually draw legend items if we want to represent the classes
    // For simplicity, we'll skip adding the lines to the legend here,
    // but mark the points.

    chart.draw_series(PointSeries::of_element(
        vec![(0.0,0.0)], // Dummy points, won't be visible due to size 0
        0,
        ShapeStyle{color:RED.to_rgba(), filled: true, stroke_width: 1},
        &|coord, size, style| {
            EmptyElement::at(coord) // Legend shape
                + Circle::new((0,0), size, style) // Legend shape
                + Text::new("Class 1", (5, 0), ("sans-serif", 12).into_font()) // Label text
        }
    ))?.label("Class 1");

    chart.draw_series(PointSeries::of_element(
        vec![(0.0,0.0)], // Dummy points
        0,
        ShapeStyle{color:BLUE.to_rgba(), filled: true, stroke_width: 1},
        &|coord, size, style| {
            EmptyElement::at(coord)
                + Circle::new((0,0), size, style)
                + Text::new("Class 0", (5, 0), ("sans-serif", 12).into_font())
        }
    ))?.label("Class 0");

    chart.configure_series_labels().draw()?;


    // Ensure all drawing operations are flushed
    root_area.present()?;
    Ok(())
}


// --- Main Function ---
fn main() {
    let seed: u64 = 42;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data.csv"); 
    // --- Load Data ---
    let (inputs, target) = match csv_reader(path) {
        Ok((inputs, target)) => (inputs, target),
        Err(e) => {
            eprintln!("Error reading CSV file: {}", e);
            process::exit(1);
        }
    };

    println!("Loaded {} samples with {} features.", inputs.nrows(), inputs.ncols());

    // --- Set Hyperparameters ---
    let learning_rate = 0.01;
    let epochs = 25;

    // --- Train the Perceptron ---
    let boundary_lines = train_perceptron_algorithm(
        &inputs,
        &target,
        learning_rate,
        epochs,
        &mut rng,
    );

    // --- Output Text Results ---
    println!("\nBoundary lines (slope, intercept) per epoch:");
    if inputs.ncols() == 2 {
        for (i, line) in boundary_lines.iter().enumerate() {
            if line.0.is_nan() || line.1.is_nan() {
                println!("Epoch {}: (Vertical Line or Error)", i + 1);
            } else {
                println!("Epoch {}: ({:.4}, {:.4})", i + 1, line.0, line.1);
            }
        }
    } else {
        println!("Boundary lines not calculated (requires 2 features).")
    }

    // --- Plot Results ---
    if inputs.ncols() == 2 { // Only plot if data is 2D
        println!("\nGenerating plot...");
        let plot_result = plot_perceptron(
            "perceptron_plot.png",
            "Perceptron Training",
            &inputs,
            &target,
            &boundary_lines
        );
        match plot_result {
            Ok(()) => println!("Plot saved to perceptron_plot.png"),
            Err(e) => eprintln!("Error generating plot: {}", e),
        }
    } else {
        println!("\nPlotting skipped (requires 2 features).")
    }
}