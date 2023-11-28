use std::{
    fmt::{Debug, Display},
    time::Duration,
};

fn map(x: f64, in_min: f64, in_max: f64, out_min: f64, out_max: f64) -> f64 {
    assert!(in_min <= x && x <= in_max);

    let x = (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;

    x
}

pub trait Histogrammable: Sized {
    fn min_max(selves: &[Self]) -> (&Self, &Self);
    fn compute_bucket<const N: usize>(&self, min: &Self, max: &Self) -> usize;
}

impl Histogrammable for f64 {
    fn min_max(selves: &[Self]) -> (&Self, &Self) {
        let min = selves
            .iter()
            .min_by(|&l, &r| Self::total_cmp(l, r))
            .unwrap();
        let max = selves
            .iter()
            .max_by(|&l, &r| Self::total_cmp(l, r))
            .unwrap();
        (min, max)
    }

    fn compute_bucket<const N: usize>(&self, min: &Self, max: &Self) -> usize {
        let idx = map(*self, *min, *max, 0.0, N as Self);
        let idx = idx as usize;
        if idx >= N {
            N - 1
        } else {
            idx
        }
    }
}

impl Histogrammable for Duration {
    fn min_max(selves: &[Self]) -> (&Self, &Self) {
        let min = selves.iter().min().unwrap();
        let max = selves.iter().max().unwrap();
        (min, max)
    }

    fn compute_bucket<const N: usize>(&self, min: &Self, max: &Self) -> usize {
        f64::compute_bucket::<N>(&self.as_secs_f64(), &min.as_secs_f64(), &max.as_secs_f64())
    }
}

#[derive(Debug, Clone)]
pub struct Histogram<const N: usize> {
    buckets: [usize; N],
    total: usize,
    highest_count: usize,
}

impl<const N: usize> Histogram<N> {
    pub fn make_with<T: Histogrammable>(values: &[T]) -> Self {
        assert!(values.len() >= N);
        let (min, max) = T::min_max(values);

        let mut buckets = [0; N];
        let mut highest_count = 0;

        for val in values {
            let idx = T::compute_bucket::<N>(val, min, max);
            buckets[idx] += 1;
            highest_count = usize::max(highest_count, buckets[idx]);
        }

        Self {
            buckets,
            total: values.len(),
            highest_count,
        }
    }
}

impl<const N: usize> Display for Histogram<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        static RESOLUTION: usize = 30;

        let mapping =
            |value: usize| -> usize { usize::div_ceil(value * RESOLUTION, self.highest_count) };

        let count_width = self.highest_count.ilog10() + 1;
        writeln!(f, "{:%^r$}", "", r = RESOLUTION + 1)?;

        for val in self.buckets {
            let mapped = mapping(val);
            assert!(mapped <= RESOLUTION);
            let rest = RESOLUTION - mapped;

            writeln!(
                f,
                "|{:*>m$}{: >r$} ({: >c$}/{} ~ {: >2}%)",
                "",
                "",
                val,
                self.total,
                (val * 100) / self.total,
                m = mapped,
                r = rest,
                c = count_width as usize
            )?;
        }
        writeln!(f, "{:%^r$}", "", r = RESOLUTION + 1)?;
        Ok(())
    }
}
