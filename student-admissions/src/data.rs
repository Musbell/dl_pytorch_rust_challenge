use anyhow::Result;
use csv::ReaderBuilder;
use rand::seq::SliceRandom;
use rand::RngCore;                     // generic RNG trait
use serde::Deserialize;
use std::collections::HashSet;
use std::iter::FromIterator;

/* ────────────────────────────── CSV record ─────────────────────────────── */
#[derive(Debug, Deserialize, Clone)]
pub struct Record {
    pub admit: i32,
    pub gre:   i32,
    pub gpa:   f64,
    pub rank:  i32,
}

#[derive(Debug, Clone)]
pub struct Admissions {
    pub rows: Vec<Record>,
}

pub trait CsvLoad {
    fn from_csv(path: &str) -> Result<Admissions>;
}

impl CsvLoad for Admissions {
    fn from_csv(path: &str) -> Result<Admissions> {
        let mut rdr = ReaderBuilder::new()
            .has_headers(true)
            .from_path(path)?;
        let rows = rdr.deserialize::<Record>().collect::<Result<_, _>>()?;
        Ok(Admissions { rows })
    }
}

/* ────────────────────────── one‑hot structures ─────────────────────────── */
#[derive(Debug, Clone)]
pub struct RecordOH {
    pub admit: i32,
    pub gre:   f64,
    pub gpa:   f64,
    pub r1:    i32,
    pub r2:    i32,
    pub r3:    i32,
    pub r4:    i32,
}

#[derive(Debug, Clone)]
pub struct AdmissionsOneHot {
    pub rows: Vec<RecordOH>,
}

/* allow `collect::<AdmissionsOneHot>()` */
impl FromIterator<RecordOH> for AdmissionsOneHot {
    fn from_iter<I: IntoIterator<Item = RecordOH>>(iter: I) -> Self {
        Self { rows: iter.into_iter().collect() }
    }
}

/* ───────────────────── AdmissionsOps for raw CSV ───────────────────────── */
pub trait AdmissionsOps {
    fn one_hot_rank(self) -> AdmissionsOneHot;
    fn head(&self, n: usize);
}

impl AdmissionsOps for Admissions {
    fn one_hot_rank(self) -> AdmissionsOneHot {
        let rows = self.rows.into_iter().map(|r| {
            let (r1, r2, r3, r4) = match r.rank {
                1 => (1, 0, 0, 0),
                2 => (0, 1, 0, 0),
                3 => (0, 0, 1, 0),
                4 => (0, 0, 0, 1),
                _ => (0, 0, 0, 0),
            };
            RecordOH { admit: r.admit, gre: r.gre as f64, gpa: r.gpa, r1, r2, r3, r4 }
        }).collect();
        AdmissionsOneHot { rows }
    }

    fn head(&self, n: usize) {
        println!("{:>5} {:>4} {:>4} {:>4}", "admit", "gre", "gpa", "rank");
        for r in self.rows.iter().take(n) {
            println!("{:>5} {:>4} {:>4} {:>4}", r.admit, r.gre, r.gpa, r.rank);
        }
    }
}

/* ───────────── AdmissionsOneHotOps (generic RNG) ───────────────────────── */
pub trait AdmissionsOneHotOps {
    fn scale_mut(&mut self);

    fn split_train_test<R>(self, test: f64, rng: &mut R)
                           -> (AdmissionsOneHot, AdmissionsOneHot)
    where
        R: RngCore + ?Sized;

    fn split_xy(&self) -> XY;
}

/// Return type of `split_xy`
pub struct XY {
    pub x: Vec<Vec<f64>>,   // Features per row (6 original + 1 bias = 7)
    pub y: Vec<i32>,        // admit labels
}

impl AdmissionsOneHotOps for AdmissionsOneHot {
    /* ---------- scaling ---------- */
    fn scale_mut(&mut self) {
        for r in &mut self.rows {
            r.gre /= 800.0;  // GRE range 200‑800 → 0‑1
            r.gpa /= 4.0;    // GPA range   1‑4  → 0‑1
        }
    }

    /* ---------- train/test split (generic RNG) ---------- */
    fn split_train_test<R>(
        self,
        test: f64,
        rng: &mut R,
    ) -> (AdmissionsOneHot, AdmissionsOneHot)
    where
        R: RngCore + ?Sized,
    {
        let total = self.rows.len();
        if total == 0 {
            return (Self { rows: vec![] }, Self { rows: vec![] });
        }

        /* shuffle indices with the provided RNG */
        let mut idx: Vec<usize> = (0..total).collect();
        idx.shuffle(rng);

        let test_cnt = ((total as f64 * test).round() as usize)
            .max(1)
            .min(total);
        let (test_idx, _) = idx.split_at(test_cnt);
        let test_set: HashSet<_> = test_idx.iter().copied().collect();

        let mut train = Vec::with_capacity(total - test_cnt);
        let mut test  = Vec::with_capacity(test_cnt);

        for (i, row) in self.rows.into_iter().enumerate() {
            if test_set.contains(&i) {
                test.push(row);
            } else {
                train.push(row);
            }
        }
        (Self { rows: train }, Self { rows: test })
    }

    /* ---------- split into X / y ---------- */
    fn split_xy(&self) -> XY {
        let mut x = Vec::with_capacity(self.rows.len());
        let mut y = Vec::with_capacity(self.rows.len());

        for r in &self.rows {
            let mut features = vec![
                r.gre,
                r.gpa,
                r.r1 as f64,
                r.r2 as f64,
                r.r3 as f64,
                r.r4 as f64,
            ];
            // Add the bias feature (constant value 1.0)
            features.push(1.0);
            x.push(features);
            y.push(r.admit);
        }
        XY { x, y }
    }
}

/* ─────────────────────── pretty‑printer for debug ──────────────────────── */
pub trait AdmissionsOneHotPrint {
    fn head_oh(&self, n: usize);
}

impl AdmissionsOneHotPrint for AdmissionsOneHot {
    fn head_oh(&self, n: usize) {
        println!("{:>5} {:>8} {:>8} {:>3} {:>3} {:>3} {:>3} {:>6}",
                 "admit", "gre", "gpa", "r1", "r2", "r3", "r4", "bias");
        for r in self.rows.iter().take(n) {
            println!("{:>5} {:>8.4} {:>8.4} {:>3} {:>3} {:>3} {:>3}",
                     r.admit, r.gre, r.gpa, r.r1, r.r2, r.r3, r.r4);
        }
    }
}