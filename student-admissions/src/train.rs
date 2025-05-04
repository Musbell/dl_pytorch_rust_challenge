// ────────────────────────────────── src/train.rs ────────────────────────────
use anyhow::Result;
use rand_distr::{Distribution, Normal};
use rand::{SeedableRng, rngs::StdRng};

use crate::data::XY;
use crate::model::{Sigmoid, Loss, Activation};

#[inline]
fn dot(x: &[f64], w: &[f64]) -> f64 {
    x.iter().zip(w).map(|(xi, wi)| xi * wi).sum()
}

#[inline]
fn predict(x: &[f64], w: &[f64]) -> f64 {
    Sigmoid::forward(dot(x, w))
}

/// Batch‑gradient–descent trainer for a single‑layer (log‑reg) network.
/// Returns the learned weight vector.
pub fn train_nn<L>(xy: &XY, epochs: usize, lr: f64) -> Result<Vec<f64>>
where
    L: Loss<Sigmoid>,
{
    /* ── weight initialisation ───────────────────────────────────────────── */
    let n_features = xy.x[0].len();
    let normal = Normal::new(0.0, 1.0 / (n_features as f64).sqrt())
        .expect("σ must be positive");

    let mut rng = StdRng::seed_from_u64(42);
    let mut w: Vec<f64> = (0..n_features)
        .map(|_| normal.sample(&mut rng))
        .collect();

    /* ── training loop ───────────────────────────────────────────────────── */
    for e in 0..epochs {
        let mut del_w = vec![0.0; n_features];

        // accumulate gradients over the whole batch
        for (x, &y) in xy.x.iter().zip(&xy.y) {
            let z    = dot(x, &w);
            let out  = Sigmoid::forward(z);
            let err  = L::delta(y as f64, out, z);

            for (dw_i, &x_i) in del_w.iter_mut().zip(x) {
                *dw_i += err * x_i;
            }
        }

        // gradient‑descent step
        for (wi, &dw_i) in w.iter_mut().zip(&del_w) {
            *wi -= lr * dw_i;
        }

        // progress log every 10 %
        if e % (epochs / 10).max(1) == 0 {
            let loss: f64 = xy
                .x
                .iter()
                .zip(&xy.y)
                .map(|(x, &y)| L::loss(y as f64, predict(x, &w)))
                .sum::<f64>()
                / xy.x.len() as f64;

            println!("epoch {e:>4} – loss {loss:.6}");
        }
    }

    Ok(w)
}
