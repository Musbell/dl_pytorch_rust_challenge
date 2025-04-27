use plotters::prelude::*;
use plotters::coord::types::RangedCoordf64;

/// Plot admitted and rejected points.
pub fn plot_points<DB: DrawingBackend>(
    chart:&mut ChartContext<DB, Cartesian2d<RangedCoordf64, RangedCoordf64>>,
    admitted: &[(f64, f64)],
    rejected: &[(f64, f64)],
) -> Result<(), DrawingAreaErrorKind<DB::ErrorType>> {
    
    chart.draw_series(
        rejected.iter().map(|&(x, y)| Circle::new((x, y), 3, RED.filled()))
    )?
    .label("Rejected (0)")
    .legend(|(x, y)| Circle::new((x, y), 3, RED.filled()));
    
    chart.draw_series(
        admitted.iter().map(|&(x, y)| Circle::new((x, y), 3, GREEN.filled()))
    )?
    .label("Admitted (1)")
        .legend(|(x, y)| Circle::new((x, y), 3, GREEN.filled()));
    Ok(())
}


/// Draw a linear decision boundary: w0*x + w1*y + b = 0.
pub fn draw_boundary_line<DB: DrawingBackend>(
    chart: &mut ChartContext<DB, Cartesian2d<RangedCoordf64, RangedCoordf64>>,
    x_range: &std::ops::Range<f64>,
    weights: &[f64],
    bias: f64,
    color: PaletteColor<Palette99>,
    label: &str,
) -> Result<(), DrawingAreaErrorKind<DB::ErrorType>> {
    if weights.len() < 2 || weights[1].abs() < 1e-8 {
        return Ok(());
    }
    let (w0, w1) = (weights[0], weights[1]);
    let y0 = (-w0 * x_range.start - bias) / w1;
    let y1 = (-w0 * x_range.end   - bias) / w1;

    chart.draw_series(LineSeries::new(
        vec![(x_range.start, y0), (x_range.end, y1)],
        color.stroke_width(2),
    ))?
        .label(label)
        .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color.stroke_width(2)));

    Ok(())
}