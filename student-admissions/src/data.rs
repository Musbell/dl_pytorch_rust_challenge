use anyhow::Result;
use csv::ReaderBuilder;
use rand::rngs::ThreadRng;
use rand::seq::SliceRandom;
use serde::Deserialize;
use std::collections::HashSet;
use std::iter::FromIterator;

/* ---------------------------------------------------------------- CSV record */

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
        let mut rdr = ReaderBuilder::new().has_headers(true).from_path(path)?;
        let rows    = rdr.deserialize::<Record>().collect::<Result<_, _>>()?;
        Ok(Admissions { rows })
    }
}

/* --------------------------------------------------------- one‑hot structures */

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

/* 🔑 allow `collect::<AdmissionsOneHot>()` */
impl FromIterator<RecordOH> for AdmissionsOneHot {
    fn from_iter<I: IntoIterator<Item = RecordOH>>(iter: I) -> Self {
        Self { rows: iter.into_iter().collect() }
    }
}

/* -------------------------------------------------------------- AdmissionsOps */

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
            RecordOH {
                admit: r.admit,
                gre:   r.gre as f64,
                gpa:   r.gpa,
                r1, r2, r3, r4,
            }
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

/* -------------------------------------------------------- AdmissionsOneHotOps */

pub trait AdmissionsOneHotOps {
    fn scale_mut(&mut self);
    fn split_train_test(self, test: f64, rng: &mut ThreadRng)
                        -> (AdmissionsOneHot, AdmissionsOneHot);
    
    /// Splitting the data into features and targets (labels)
    // Split the one‑hot data into
    ///   * `x` – features  `[gre, gpa, r1, r2, r3, r4]`
    ///   * `y` – target    `admit`
    fn split_xy(&self) -> XY;
}

/// Convenience container returned by `split_xy`.
pub struct XY {
    /// Each inner `Vec<f64>` holds the 6 feature values for one row
    /// (scaled gre, scaled gpa, r1 … r4).
    pub x: Vec<Vec<f64>>,
    /// Target labels (0 = rejected, 1 = admitted)
    pub y: Vec<i32>,
}

impl AdmissionsOneHotOps for AdmissionsOneHot {
    fn scale_mut(&mut self) {
        if self.rows.is_empty() { return; }

        let len     = self.rows.len() as f64;
        let (sum_gre, sum_gpa) = self.rows.iter()
            .fold((0.0, 0.0), |(sgre, sgpa), r| (sgre + r.gre, sgpa + r.gpa));
        let mean_gre = sum_gre / len;
        let mean_gpa = sum_gpa / len;

        let (var_gre, var_gpa) = self.rows.iter()
            .fold((0.0, 0.0), |(vg, vp), r| {
                (vg + (r.gre - mean_gre).powi(2),
                 vp + (r.gpa - mean_gpa).powi(2))
            });
        let sd_gre = (var_gre / len).sqrt().max(1.0);
        let sd_gpa = (var_gpa / len).sqrt().max(1.0);

        for r in &mut self.rows {
            r.gre = (r.gre - mean_gre) / sd_gre;
            r.gpa = (r.gpa - mean_gpa) / sd_gpa;
        }
    }

    fn split_train_test(
        self,
        test: f64,
        rng: &mut ThreadRng,
    ) -> (AdmissionsOneHot, AdmissionsOneHot) {
        let total = self.rows.len();
        if total == 0 {
            return (Self { rows: vec![] }, Self { rows: vec![] });
        }
        let mut idx: Vec<usize> = (0..total).collect();
        idx.shuffle(rng);

        let test_cnt = ((total as f64 * test).round() as usize).max(1).min(total);
        let (test_idx, train_idx) = idx.split_at(test_cnt);
        let test_set: HashSet<_>   = test_idx.iter().copied().collect();

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

    /// Splitting the data into features and targets (labels)
    fn split_xy(&self) -> XY {
        let mut x = Vec::with_capacity(self.rows.len());
        let mut y = Vec::with_capacity(self.rows.len());

        for r in &self.rows {
            // 6 features in the same order every time
            x.push(vec![r.gre, r.gpa, r.r1 as f64, r.r2 as f64, r.r3 as f64, r.r4 as f64]);
            y.push(r.admit);
        }
        XY { x, y }
    }

}

/* -------------------------------------------------------------- pretty‑printer */

pub trait AdmissionsOneHotPrint {
    fn head_oh(&self, n: usize);
}

impl AdmissionsOneHotPrint for AdmissionsOneHot {
    fn head_oh(&self, n: usize) {
        println!(
            "{:>5} {:>8} {:>8} {:>3} {:>3} {:>3} {:>3}",
            "admit", "gre", "gpa", "r1", "r2", "r3", "r4"
        );
        for r in self.rows.iter().take(n) {
            println!(
                "{:>5} {:>8.4} {:>8.4} {:>3} {:>3} {:>3} {:>3}",
                r.admit, r.gre, r.gpa, r.r1, r.r2, r.r3, r.r4
            );
        }
    }
}
