use anyhow::Result;          // ? sugar + Send/Sync ready
use plotters::prelude::*;

use crate::data::RecordOH;

/// Scatter‑plot scaled one‑hot data
pub fn plot_admissions_oh(
    records: &[RecordOH],
    png: &str,
    title: &str,
) -> Result<()> {           
    /* 1 ── drawing area */
    let root = BitMapBackend::new(png, (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;

    /* 2 ── ranges */
    let (min_gre, max_gre) = records.iter().fold(
        (f64::INFINITY, f64::NEG_INFINITY),
        |(lo, hi), r| (lo.min(r.gre), hi.max(r.gre)),
    );
    let (min_gpa, max_gpa) = records.iter().fold(
        (f64::INFINITY, f64::NEG_INFINITY),
        |(lo, hi), r| (lo.min(r.gpa), hi.max(r.gpa)),
    );
    let gre = (min_gre - 0.5)..(max_gre + 0.5);
    let gpa = (min_gpa - 0.5)..(max_gpa + 0.5);

    /* 3 ── chart context */
    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 20).into_font())
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(gre, gpa)?;

    chart.configure_mesh()
        .x_desc("Test (GRE) – scaled")
        .y_desc("Grades (GPA) – scaled")
        .bold_line_style(&BLACK.mix(0.3))
        .light_line_style(&BLACK.mix(0.15))
        .draw()?;

    /* 4 ── data points */
    let rejected: Vec<_> =
        records.iter().filter(|r| r.admit == 0).map(|r| (r.gre, r.gpa)).collect();
    let admitted: Vec<_> =
        records.iter().filter(|r| r.admit == 1).map(|r| (r.gre, r.gpa)).collect();

    chart.draw_series(
        rejected.iter().map(|&(x, y)| Circle::new((x, y), 4, RED.filled())),
    )?;
    chart.draw_series(
        admitted.iter().map(|&(x, y)| Circle::new((x, y), 4, CYAN.filled())),
    )?;

    let outline = BLACK.stroke_width(1);
    chart.draw_series(
        rejected.iter().map(|&(x, y)| Circle::new((x, y), 4, outline.clone())),
    )?;
    chart.draw_series(
        admitted.iter().map(|&(x, y)| Circle::new((x, y), 4, outline.clone())),
    )?;

    root.present()?;
    Ok(())
}
