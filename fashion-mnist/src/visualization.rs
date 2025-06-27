use plotters::prelude::*;
use burn::data::dataset::vision::MnistItem;
use burn::data::dataset::Dataset;
use std::path::Path;
use std::error::Error;
use std::fs;

use crate::dataset::FashionMNISTDataset;
use crate::model::Network;
use crate::train::get_class_name;
use burn::prelude::*;

// Constants for visualization
const OUTPUT_DIR: &str = "./fashion-mnist/visualizations";
const IMAGE_SIZE: u32 = 28;
const PLOT_SIZE: u32 = 200;
const GRID_SIZE: u32 = 5;

/// Ensure the output directory exists
fn ensure_output_dir() -> std::io::Result<()> {
    fs::create_dir_all(OUTPUT_DIR)
}

/// Visualize a single Fashion MNIST image
pub fn visualize_image(item: &MnistItem, filename: &str) -> Result<(), Box<dyn Error>> {
    ensure_output_dir()?;
    
    let path = Path::new(OUTPUT_DIR).join(filename);
    let root = BitMapBackend::new(&path, (PLOT_SIZE, PLOT_SIZE)).into_drawing_area();
    root.fill(&WHITE)?;
    
    // Create a drawing area for the image
    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!("Label: {} ({})", get_class_name(item.label as i32), item.label),
            ("sans-serif", 15),
        )
        .build_cartesian_2d(0..IMAGE_SIZE, 0..IMAGE_SIZE)?;
    
    chart.configure_mesh().disable_mesh().draw()?;
    
    // Draw the image pixel by pixel using PointSeries
    for y in 0..IMAGE_SIZE {
        for x in 0..IMAGE_SIZE {
            let pixel_value = item.image[y as usize][x as usize];
            let color = RGBColor(
                (255.0 - pixel_value) as u8,
                (255.0 - pixel_value) as u8,
                (255.0 - pixel_value) as u8,
            );
            
            // Draw a single point for each pixel
            chart.draw_series(std::iter::once(Circle::new(
                (x, y),
                1,
                color.filled(),
            )))?;
        }
    }
    
    root.present()?;
    println!("Image saved to {}", path.display());
    
    Ok(())
}

/// Visualize a grid of Fashion MNIST images
pub fn visualize_image_grid(dataset: &FashionMNISTDataset, filename: &str) -> Result<(), Box<dyn Error>> {
    ensure_output_dir()?;
    
    let path = Path::new(OUTPUT_DIR).join(filename);
    let root = BitMapBackend::new(&path, (PLOT_SIZE * GRID_SIZE, PLOT_SIZE * GRID_SIZE)).into_drawing_area();
    root.fill(&WHITE)?;
    
    // Split the drawing area into a grid
    let areas = root.split_evenly((GRID_SIZE as usize, GRID_SIZE as usize));
    
    for (i, area) in areas.iter().enumerate() {
        if i >= GRID_SIZE as usize * GRID_SIZE as usize {
            break;
        }
        
        if let Some(item) = dataset.get(i) {
            let mut chart = ChartBuilder::on(area)
                .caption(
                    format!("{} ({})", get_class_name(item.label as i32), item.label),
                    ("sans-serif", 10),
                )
                .build_cartesian_2d(0..IMAGE_SIZE, 0..IMAGE_SIZE)?;
            
            chart.configure_mesh().disable_mesh().draw()?;
            
            // Collect all points for this image
            let mut points = Vec::new();
            for y in 0..IMAGE_SIZE {
                for x in 0..IMAGE_SIZE {
                    let pixel_value = item.image[y as usize][x as usize];
                    if pixel_value > 0.0 {  // Only draw non-zero pixels
                        let intensity = (255.0 - pixel_value) as u8;
                        points.push((x, y, RGBColor(intensity, intensity, intensity)));
                    }
                }
            }
            
            // Draw all points at once
            chart.draw_series(points.iter().map(|(x, y, color)| {
                Rectangle::new([(*x, *y), (*x + 1, *y + 1)], color.filled())
            }))?;
        }
    }
    
    root.present()?;
    println!("Image grid saved to {}", path.display());
    
    Ok(())
}

/// Visualize training loss over epochs
pub fn visualize_training_loss(losses: &[f32], filename: &str) -> Result<(), Box<dyn Error>> {
    ensure_output_dir()?;
    
    let path = Path::new(OUTPUT_DIR).join(filename);
    let root = BitMapBackend::new(&path, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;
    
    let max_loss = losses.iter().fold(0.0f32, |a, &b| a.max(b));
    let min_loss = losses.iter().fold(f32::MAX, |a, &b| a.min(b));
    let padding = (max_loss - min_loss) * 0.1;
    
    let mut chart = ChartBuilder::on(&root)
        .caption("Training Loss", ("sans-serif", 30))
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(40)
        .build_cartesian_2d(
            0..losses.len(),
            (min_loss - padding)..(max_loss + padding),
        )?;
    
    chart
        .configure_mesh()
        .x_labels(losses.len())
        .x_desc("Epoch")
        .y_desc("Loss")
        .draw()?;
    
    chart.draw_series(LineSeries::new(
        losses.iter().enumerate().map(|(i, &loss)| (i, loss)),
        &RED,
    ))?;
    
    root.present()?;
    println!("Training loss plot saved to {}", path.display());
    
    Ok(())
}

/// Visualize model predictions compared to actual labels
pub fn visualize_predictions<B: Backend>(
    model: &Network<B>,
    dataset: &FashionMNISTDataset,
    device: &B::Device,
    filename: &str,
) -> Result<(), Box<dyn Error>> {
    ensure_output_dir()?;
    
    let path = Path::new(OUTPUT_DIR).join(filename);
    let root = BitMapBackend::new(&path, (PLOT_SIZE * GRID_SIZE, PLOT_SIZE * GRID_SIZE)).into_drawing_area();
    root.fill(&WHITE)?;
    
    // Split the drawing area into a grid
    let areas = root.split_evenly((GRID_SIZE as usize, GRID_SIZE as usize));
    
    for (i, area) in areas.iter().enumerate() {
        if i >= GRID_SIZE as usize * GRID_SIZE as usize {
            break;
        }
        
        if let Some(item) = dataset.get(i) {
            // Create a tensor from the image
            let image_data: Vec<f32> = item.image.iter()
                .flat_map(|row| row.iter().cloned())
                .collect();
            
            // Normalize the image data
            let normalized_data: Vec<f32> = image_data.iter()
                .map(|&pixel| (pixel / 255.0 - 0.1307) / 0.3081)
                .collect();
            
            // Create a tensor and reshape it for the model
            let tensor_data = TensorData::new(normalized_data, [1, 28, 28]);
            let image_tensor = Tensor::<B, 3>::from_data(tensor_data.convert::<B::FloatElem>(), device);
            let flattened_tensor = image_tensor.reshape([1, 784]);
            
            // Get model prediction
            let output = model.forward(flattened_tensor);
            let predictions = output.argmax(1);
            let predicted_label: i32 = predictions.to_data().into_vec().unwrap()[0];
            
            // Draw the image with prediction information
            let mut chart = ChartBuilder::on(area)
                .caption(
                    format!(
                        "Pred: {} ({}), True: {} ({})",
                        get_class_name(predicted_label),
                        predicted_label,
                        get_class_name(item.label as i32),
                        item.label
                    ),
                    ("sans-serif", 8),
                )
                .build_cartesian_2d(0..IMAGE_SIZE, 0..IMAGE_SIZE)?;
            
            chart.configure_mesh().disable_mesh().draw()?;
            
            // Collect all points for this image
            let mut points = Vec::new();
            for y in 0..IMAGE_SIZE {
                for x in 0..IMAGE_SIZE {
                    let pixel_value = item.image[y as usize][x as usize];
                    if pixel_value > 0.0 {  // Only draw non-zero pixels
                        // Use different colors for correct and incorrect predictions
                        let color = if predicted_label == item.label as i32 {
                            let intensity = (255.0 - pixel_value) as u8;
                            RGBColor(intensity, intensity, intensity)
                        } else {
                            // Add a red tint to incorrect predictions
                            let intensity = (255.0 - pixel_value) as u8;
                            RGBColor(255, intensity, intensity)
                        };
                        points.push((x, y, color));
                    }
                }
            }
            
            // Draw all points at once
            chart.draw_series(points.iter().map(|(x, y, color)| {
                Rectangle::new([(*x, *y), (*x + 1, *y + 1)], color.filled())
            }))?;
        }
    }
    
    root.present()?;
    println!("Prediction visualization saved to {}", path.display());
    
    Ok(())
}

/// Visualize confusion matrix
pub fn visualize_confusion_matrix<B: Backend>(
    model: &Network<B>,
    dataset: &FashionMNISTDataset,
    device: &B::Device,
    filename: &str,
) -> Result<(), Box<dyn Error>> {
    ensure_output_dir()?;
    
    let path = Path::new(OUTPUT_DIR).join(filename);
    let root = BitMapBackend::new(&path, (800, 800)).into_drawing_area();
    root.fill(&WHITE)?;
    
    // Initialize confusion matrix
    let mut confusion_matrix = [[0u32; 10]; 10];
    
    // Calculate confusion matrix
    let num_samples = std::cmp::min(dataset.len(), 1000); // Limit to 1000 samples for speed
    for i in 0..num_samples {
        if let Some(item) = dataset.get(i) {
            // Create a tensor from the image
            let image_data: Vec<f32> = item.image.iter()
                .flat_map(|row| row.iter().cloned())
                .collect();
            
            // Normalize the image data
            let normalized_data: Vec<f32> = image_data.iter()
                .map(|&pixel| (pixel / 255.0 - 0.1307) / 0.3081)
                .collect();
            
            // Create a tensor and reshape it for the model
            let tensor_data = TensorData::new(normalized_data, [1, 28, 28]);
            let image_tensor = Tensor::<B, 3>::from_data(tensor_data.convert::<B::FloatElem>(), device);
            let flattened_tensor = image_tensor.reshape([1, 784]);
            
            // Get model prediction
            let output = model.forward(flattened_tensor);
            let predictions = output.argmax(1);
            let predicted_label: i32 = predictions.to_data().into_vec().unwrap()[0];
            
            // Update confusion matrix
            confusion_matrix[item.label as usize][predicted_label as usize] += 1;
        }
    }
    
    // Find the maximum value in the confusion matrix for color scaling
    let max_value = confusion_matrix.iter()
        .flat_map(|row| row.iter())
        .fold(0, |max, &val| max.max(val));
    
    // Create the chart with f64 ranges
    let mut chart = ChartBuilder::on(&root)
        .caption("Confusion Matrix", ("sans-serif", 30))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0f64..10f64, 0f64..10f64)?;
    
    chart
        .configure_mesh()
        .x_labels(10)
        .y_labels(10)
        .x_desc("Predicted Label")
        .y_desc("True Label")
        .x_label_formatter(&|x| get_class_name(*x as i32).to_string())
        .y_label_formatter(&|y| get_class_name(*y as i32).to_string())
        .draw()?;
    
    // Draw the confusion matrix
    for (true_label, row) in confusion_matrix.iter().enumerate() {
        for (pred_label, &count) in row.iter().enumerate() {
            if count > 0 {
                // Calculate color intensity based on count
                let intensity = (count as f64 / max_value as f64 * 255.0) as u8;
                let color = RGBColor(255 - intensity, 255 - intensity, 255);
                
                // Draw a rectangle for each cell
                chart.draw_series(std::iter::once(Rectangle::new(
                    [(pred_label as f64, true_label as f64), ((pred_label + 1) as f64, (true_label + 1) as f64)],
                    color.filled(),
                )))?;
                
                // Add text with the count
                chart.draw_series(std::iter::once(Text::new(
                    count.to_string(),
                    (pred_label as f64 + 0.5, true_label as f64 + 0.5),
                    ("sans-serif", 15).into_font().color(&BLACK),
                )))?;
            }
        }
    }
    
    root.present()?;
    println!("Confusion matrix saved to {}", path.display());
    
    Ok(())
}