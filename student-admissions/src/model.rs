// ────────────────── 1.  Activation trait ──────────────────
pub trait Activation {
    /// σ(x)
    fn forward(x: f64) -> f64;
    /// σ′(x)
    fn derivative(x: f64) -> f64;
}

/// Sigmoid activation σ(x) = 1 / (1 + e^(−x))
pub struct Sigmoid;

impl Activation for Sigmoid {
    #[inline]
    fn forward(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    #[inline]
    fn derivative(x: f64) -> f64 {
        let s = Self::forward(x);
        s * (1.0 - s)
    }
}

// ────────────────── 2.  Loss trait ──────────────────
pub trait Loss<A: Activation> {
    /// ℒ(y, ŷ)
    fn loss(y: f64, output: f64) -> f64;

    /// δ (error term) for back‑prop at the output neuron.
    ///
    /// `pre_act` is z, the value **before** the activation.
    fn delta(y: f64, output: f64, pre_act: f64) -> f64;
}

/// Binary‑cross‑entropy (log loss)
pub struct BCE;

impl Loss<Sigmoid> for BCE {
    #[inline]
    fn loss(y: f64, output: f64) -> f64 {
        // clamp for ln‑safety if desired
        -(y * output.ln() + (1.0 - y) * (1.0 - output).ln())
    }

    #[inline]
    fn delta(y: f64, output: f64, _z: f64) -> f64 {
        // For sigmoid+BCE, σ′ cancels ⇒ δ = ŷ − y
        output - y
    }
}

/// Mean‑squared error ½(ŷ − y)²
pub struct MSE;

impl Loss<Sigmoid> for MSE {
    #[inline]
    fn loss(y: f64, output: f64) -> f64 {
        0.5 * (output - y).powi(2)
    }

    #[inline]
    fn delta(y: f64, output: f64, z: f64) -> f64 {
        // δ = (ŷ − y) * σ′(z)
        (output - y) * Sigmoid::derivative(z)
    }
}
