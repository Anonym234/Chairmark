//! Crate for benchmarking
//!
//! Unfortunately, there's no stable support for proper `cargo bench` stuff.
//! Therefore, I made this crate
//!
//! The idea is to have a function that benchmarks another given function.
//! This will be done a specified number of times for more reliable data.
//! During this benchmark ("chairmark") running time data is collected that can be aggregated etc..
//!
//! This data can then be displayed easliy and readable.
//! There's also an additional feature for comparing different chairmarks.
//! This can be used to compare a custom implementation with one from the standard library or different implmenentations.
//!
//! # Examples
//!
//! ## example without using macros
//! ```rust
//! use chairmark::{chair, Time, Comparison};
//!
//! // checks if number is power of two
//! fn is_power(arg: u32) -> bool {
//!     arg.to_ne_bytes().into_iter().sum::<u8>() == 1
//! }
//!
//! fn main() {
//!     const NUMBER: u32 = 69;
//!
//!     // benchmark std function
//!     let std = chair(1_000, || NUMBER.is_power_of_two()).aggregate::<Time>();
//!     // benchmark custom function
//!     let custom = chair(1_000, || is_power(NUMBER)).aggregate::<Time>();
//!
//!     // compare
//!     let compare = Comparison::from([("std", std), ("custom", custom)]);
//!
//!     // display as a table
//!     println!("{}", compare);
//! }
//! ```
//!
//! ## example with all macros
//! ```rust
//! use chairmark::*;
//!
//! // checks if number is power of two
//! fn is_power(arg: u32) -> bool {
//!     arg.to_ne_bytes().into_iter().sum::<u8>() == 1
//! }
//!
//! fn main() {
//!     const NUMBER: u32 = 69;
//!
//!     // benchmark std function
//!     let std = chair(1_000, || NUMBER.is_power_of_two());
//!     // benchmark custom function
//!     let custom = chair(1_000, || is_power(NUMBER));
//!
//!     // compare
//!     let compare = agg_and_cmp![std, custom];
//!
//!     // display as table
//!     println!("{}", compare);
//! }
//! ```
//!
//! # example with prepared data
//! ```rust
//! use chairmark::*;
//!
//! // custom sort function
//! fn bubblesort(data: &mut Vec<u32>) {
//!     /* your great bubblesort implementation */
//! }
//!
//! fn main() {
//!     let prepare = |idx| (0..idx).map(|x| x as u32 / 3).collect::<Vec<_>>();
//!
//!     // benchmark std functions
//!     let std_stable = chair_prepare(1_000, prepare, |mut data| data.sort());
//!     let std_unstable = chair_prepare(1_000, prepare, |mut data| data.sort_unstable());
//!     // benchmark custom function
//!     let custom = chair_prepare(1_000, prepare, |mut data| bubblesort(&mut data));
//!
//!     // aggregate and compare
//!     let compare = agg_and_cmp![std_stable, std_unstable, custom];
//!
//!     println!("{}", compare);
//! }
//! ```

use mdtable::{Builder, Table};
use std::{
    fmt::Display,
    time::{Duration, Instant},
};

mod histogram;
use histogram::Histogram;

/// aggregates to an [`Aggregate`] of [`Time`]
#[macro_export]
macro_rules! agg {
    ($x:ident) => {
        $x.aggregate::<Time>()
    };
}

/// instantiates a [`Comparison`] of the given values
///
/// names stringified from variable names
#[macro_export]
macro_rules! compare {
    ($($x:ident),* $(,)?) => {
        Comparison::from([ $(( stringify!($x), $x )),* ])
    };
}

/// wrappers "calls" to [`agg!`] and [`compare!`]
#[macro_export]
macro_rules! agg_and_cmp {
    ($($x:ident),* $(,)?) => {
        Comparison::from([ $(( stringify!($x), agg!($x) )),* ])
    };
}

/// timer type to time anything
///
/// usually the [`Timer::time`] function will be used with a closure, but it can be used manually
#[derive(Debug, Clone)]
struct Timer {
    start: Instant,
}

impl Timer {
    /// starts the timer and gives a new instance
    fn start() -> Self {
        Self {
            start: Instant::now(),
        }
    }

    /// stops the timer, giving the elapsed time
    ///
    /// note that time is kept using the [`std::time::Instant`] clock
    fn stop(&self) -> Duration {
        Instant::now() - self.start
    }

    /// time the given closure / function
    fn time<T>(f: impl FnOnce() -> T) -> Duration {
        let timer = Self::start();
        f();
        timer.stop()
    }
}

/// keep track of time measurements
///
/// this structure stores all inserted points and is intended to be used for multiple runs of benchmarks
///
/// it provides facilities to aggregate the given data, see `[Self::aggregate]`
#[derive(Debug, Clone)]
pub struct Measurements {
    durations: Vec<Duration>,
    is_sorted: bool,
}

impl Measurements {
    /// new, empty measurements
    fn new() -> Self {
        Self {
            durations: Vec::new(),
            is_sorted: true,
        }
    }

    /// accept a new data point (duration)
    fn accept(&mut self, duration: Duration) {
        self.durations.push(duration);
        self.is_sorted = false;
    }

    /// get number of datapoints stored
    pub fn len(&self) -> usize {
        self.durations.len()
    }

    /// checks whether no datapoints are stored
    pub fn is_empty(&self) -> bool {
        self.durations.is_empty()
    }

    /// shortest duration of collected data
    ///
    /// # panic
    /// panis if `self.is_empty()` or `!self.is_sorted`
    pub fn min(&self) -> Duration {
        assert!(self.is_sorted);
        self.durations[0]
    }

    /// longest duration of collected data
    ///
    /// # panic
    /// panis if `self.is_empty()` or `!self.is_sorted`
    pub fn max(&self) -> Duration {
        assert!(self.is_sorted);
        self.durations[self.len() - 1]
    }

    /// mean duration of collected data (arithmetic mean)
    ///
    /// # panic
    /// panis if `self.is_empty()`
    pub fn arith_mean(&self) -> Duration {
        let sum: Duration = self.durations.iter().sum();
        sum / self.len() as u32
    }

    /// median duration of collected data
    ///
    /// **NOTE**: right-biased in an even length data collection
    ///
    /// # panic
    /// panis if `self.is_empty()` or `!self.is_sorted`
    pub fn median(&self) -> Duration {
        assert!(self.is_sorted);
        *self.durations.iter().take(self.len() / 2).last().unwrap()
    }

    /// variance in duration of collected data
    ///
    /// # panic
    /// panis if `self.is_empty()`
    pub fn variance(&self) -> Duration {
        let mean = self.arith_mean();
        Duration::from_secs_f64(self.durations.iter().fold(0.0, |mut acc, duration| {
            let diff = duration.as_secs_f64() - mean.as_secs_f64();
            acc += diff * diff;
            acc
        }))
    }

    /// sorts the duration data
    ///
    /// is required before some aggregation steps
    pub fn sort(&mut self) {
        self.durations.sort_unstable();
        self.is_sorted = true;
    }

    /// aggregate all other information into one struct
    ///
    /// # panics
    /// since it uses the other aggregate functions, if the collection is empty or it's not sorted, a `panic!` is invoked
    pub fn aggregate<T>(&self) -> Aggregate<T>
    where
        Duration: Into<T>,
    {
        Aggregate {
            min: self.min().into(),
            max: self.max().into(),
            arith_mean: self.arith_mean().into(),
            median: self.median().into(),
            variance: self.variance().into(),
        }
    }

    pub fn histogram<const N: usize>(&self) -> Histogram<N> {
        Histogram::make_with(&self.durations[..])
    }
}

/// store aggregated data from [`Measurements`]
///
/// stores stuff like min, max, mean, median, variance
#[derive(Debug, Clone)]
pub struct Aggregate<T> {
    min: T,
    max: T,
    arith_mean: T,
    median: T,
    variance: T,
}

impl<T> Aggregate<T> {
    /// getter functions and names for all stored data
    const GET_DATA: [(&'static str, fn(&Self) -> &T); 5] = [
        ("min", Self::min),
        ("max", Self::max),
        ("arith_mean", Self::arith_mean),
        ("median", Self::median),
        ("variance", Self::variance),
    ];

    /// get minimum
    pub fn min(&self) -> &T {
        &self.min
    }

    /// get maximum
    pub fn max(&self) -> &T {
        &self.max
    }

    /// get arithmetic mean
    pub fn arith_mean(&self) -> &T {
        &self.arith_mean
    }

    /// get median
    pub fn median(&self) -> &T {
        &self.median
    }

    /// get variance
    pub fn variance(&self) -> &T {
        &self.variance
    }

    /// get all data with names
    ///
    /// returns an array with tuples (name, value)
    pub fn all(&self) -> [(&'static str, &T); 5] {
        Self::GET_DATA.map(|(name, getter)| (name, getter(self)))
    }
}

/// get a time aggregate from duration aggregate
impl From<Aggregate<Duration>> for Aggregate<Time> {
    fn from(value: Aggregate<Duration>) -> Self {
        Self {
            min: value.min.into(),
            max: value.max.into(),
            arith_mean: value.arith_mean.into(),
            median: value.median.into(),
            variance: value.variance.into(),
        }
    }
}

impl<T: Display> Display for Aggregate<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{{")?;
        for (name, value) in self.all() {
            writeln!(f, "  {}: {}", name, value)?;
        }
        write!(f, "}}")?;
        Ok(())
    }
}

/// custom time storing other than [`Duration`]
///
/// stores time values separate, most useful for displaying
#[derive(Debug, Clone, Copy)]
pub struct Time {
    hours: u64,
    minutes: u8,
    secs: u8,
    millis: u16,
    micros: u16,
    nanos: u16,
}

impl Time {
    /// getter functions (all as [`u64`]) and names
    const GET_UNITS: [(fn(&Self) -> u64, &'static str); 6] = [
        (Self::hours::<u64>, "h"),
        (Self::minutes::<u64>, "m"),
        (Self::secs::<u64>, "s"),
        (Self::millis::<u64>, "ms"),
        (Self::micros::<u64>, "µs"),
        (Self::nanos::<u64>, "ns"),
    ];

    /// get hours only
    pub fn hours<T: From<u64>>(&self) -> T {
        self.hours.into()
    }

    /// get minutes only
    pub fn minutes<T: From<u8>>(&self) -> T {
        self.minutes.into()
    }

    /// get seconds only
    pub fn secs<T: From<u8>>(&self) -> T {
        self.secs.into()
    }

    /// get milliseconds only
    pub fn millis<T: From<u16>>(&self) -> T {
        self.millis.into()
    }

    /// get microseconds only
    pub fn micros<T: From<u16>>(&self) -> T {
        self.micros.into()
    }

    /// get nanoseconds only
    pub fn nanos<T: From<u16>>(&self) -> T {
        self.nanos.into()
    }

    /// get total nanoseconds as an [`f64`]
    pub fn total_nanos_f64(&self) -> f64 {
        let mut total = 0.0;

        total += self.hours::<u128>() as f64;

        total *= 60.0;
        total += self.minutes::<f64>();

        total *= 60.0;
        total += self.secs::<f64>();

        total *= 1000.0;
        total += self.millis::<f64>();

        total *= 1000.0;
        total += self.micros::<f64>();

        total *= 1000.0;
        total += self.nanos::<f64>();

        total
    }
}

/// convert a [`Duration`] to [`Time`]
impl From<Duration> for Time {
    fn from(value: Duration) -> Self {
        Self::from(&value)
    }
}

/// convert a reference to a [`Duration`] into [`Time`]
impl From<&Duration> for Time {
    fn from(value: &Duration) -> Self {
        // >= secs
        let secs = value.as_secs();

        let hours = secs / 3600;
        let minutes = (secs % 3600) / 60;
        let secs = secs % 60;

        // < secs
        let nanos = value.subsec_nanos();
        let millis = nanos / 1_000_000;
        let micros = (nanos % 1_000_000) / 1_000;
        let nanos = nanos % 1_000;

        Self {
            hours,
            minutes: minutes as u8,
            secs: secs as u8,
            millis: millis as u16,
            micros: micros as u16,
            nanos: nanos as u16,
        }
    }
}

/// convert reference into Time, weird wrapper for implementing `Into<Time> for &Time`
impl<'a> From<&'a Time> for Time {
    fn from(value: &'a Time) -> Self {
        value.clone()
    }
}

/// displays the time in a good format, e.g. `1h26m50s420ms10µs333ns`
impl Display for Time {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut anything = false;
        for (getter, unit) in Self::GET_UNITS {
            let value = getter(self);
            if value > 0 {
                anything = true;
                write!(f, "{}{}", value, unit)?;
            }
        }
        if !anything {
            write!(f, "0s")?;
        }
        Ok(())
    }
}

/// comparison of multiple aggregates
///
/// most useful for displaing a good table with relative
/// (takes first element as baseline)
#[derive(Debug, Clone)]
pub struct Comparison<A, T, const N: usize>([(A, Aggregate<T>); N]);

/// convert an array of measurements into a comparison of aggregates (uses [`Duration`])
impl<A, const N: usize> From<[(A, Measurements); N]> for Comparison<A, Duration, N>
where
    A: AsRef<str>,
{
    fn from(value: [(A, Measurements); N]) -> Self {
        Self(value.map(|(name, measurements)| (name, measurements.aggregate())))
    }
}

/// convert an array of aggregates into comparison of these
impl<A, T, X, const N: usize> From<[(A, Aggregate<T>); N]> for Comparison<A, X, N>
where
    A: AsRef<str>,
    Aggregate<T>: Into<Aggregate<X>>,
{
    fn from(value: [(A, Aggregate<T>); N]) -> Self {
        Self(value.map(|(name, agg)| (name, agg.into())))
    }
}

/// displays the comparison as a (markdown) table
///
/// uses first element as baseline, other ones get a relative displayed
impl<A, T, const N: usize> Display for Comparison<A, T, N>
where
    A: AsRef<str>,
    for<'a> &'a T: Into<Time>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        static DECIMALS: i32 = 0;

        write!(f, "metric")?;
        for (name, _) in &self.0 {
            write!(f, " | {}", name.as_ref())?;
        }
        writeln!(f)?;

        write!(f, "---")?;
        for _ in 0..N {
            write!(f, " | ---:")?;
        }
        writeln!(f)?;

        for (name, getter) in Aggregate::GET_DATA {
            write!(f, "{}", name)?;

            let mut baseline: Option<Time> = None;

            for (_, agg) in &self.0 {
                let time: Time = getter(agg).into();
                write!(f, " | {}", time)?;

                if let Some(baseline) = baseline {
                    let diff_percent = {
                        let mut diff = time.total_nanos_f64() / baseline.total_nanos_f64();
                        diff -= 1.0;
                        diff *= 100.0; // make percent
                        diff *= f64::powi(10.0, DECIMALS);
                        diff = diff.floor();
                        diff *= f64::powi(10.0, -DECIMALS);
                        diff
                    };
                    write!(f, " & ({}%)", diff_percent)?;
                } else {
                    baseline = Some(time);
                }
            }

            writeln!(f)?;
        }

        Ok(())
    }
}

impl<A, T, const N: usize> Comparison<A, T, N>
where
    A: AsRef<str>,
    for<'a> &'a T: Into<Time>,
{
    /// generate a table for the comparison
    pub fn table<'x>(&'x self) -> Table<&'static str, &'x str, String, N> {
        let mut builder = Builder::new();

        builder.header(("", {
            let mut refs = [""; N];
            for i in 0..N {
                refs[i] = self.0[i].0.as_ref();
            }
            refs
        }));

        builder.default_alignments();

        for (name, getter) in Aggregate::GET_DATA {
            let mut content = Vec::with_capacity(N);

            let mut baseline: Option<Time> = None;
            for (_, agg) in &self.0 {
                let duration = getter(agg);
                let time: Time = duration.into();

                if let Some(baseline) = baseline {
                    let diff_percent = {
                        let mut diff = time.total_nanos_f64() / baseline.total_nanos_f64();
                        diff -= 1.0;
                        diff *= 100.0; // make percent
                        diff.trunc()
                    };
                    content.push(format!("{} ({:>3}%)", time, diff_percent));
                } else {
                    content.push(time.to_string());
                    baseline = Some(time);
                }
            }

            builder.row((name, content.try_into().ok().unwrap()));
        }

        builder.finish()
    }
}

/// runs a chairmark on a single function
///
/// It executes the function `runs` times, returning the measurements.
/// These [`Measurements`] can be aggregated etc..
///
/// # Example
/// ```rust
/// use chairmark::{chair, Time};
///
/// fn to_chairmark() {
///     /* some lengthy function */
/// }
///
/// fn main() {
///     let measure = chair(1_000, to_chairmark);
///     println!("{}", measure.aggregate::<Time>());
///     let measure = chair(1_000, || () /* closures work too! */);
///     println!("{}", measure.aggregate::<Time>());
/// }
/// ```
pub fn chair<Return>(runs: usize, f: impl Fn() -> Return) -> Measurements {
    chair_prepare(runs, |_| (), |_| f())
}

/// runs a chairmark with prepared data
///
/// the `prepare` function is called for every run with the current run number as argument
///
/// # Example
/// ```rust
/// use chairmark::{chair_prepare, Time};
///
/// fn prepare(run_id: usize) -> Vec<u32> {
///     (0..run_id as u32).collect()
/// }
///
/// fn main() {
///     let agg = chair_prepare(1_000, prepare, |mut data| data.sort()).aggregate::<Time>();
///     println!("{}", agg);
/// }
/// ```
pub fn chair_prepare<Data, Return>(
    runs: usize,
    prepare: impl Fn(usize) -> Data,
    f: impl Fn(Data) -> Return,
) -> Measurements {
    let mut measurements = Measurements::new();
    for i in 0..runs {
        let data = prepare(i);
        let f = &f;
        measurements.accept(Timer::time(move || f(data)));
    }

    measurements.sort();
    measurements
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table() {
        fn bubblesort(data: &mut Vec<u32>) {
            let len = data.len();
            let mut sorted = false;
            while !sorted {
                sorted = true;
                for i in 1..len {
                    let j = i - 1;
                    if data[j] < data[i] {
                        data.swap(i, j);
                        sorted = false;
                    }
                }
            }
        }

        const RUNS: usize = 1_000;
        let prepare = |_| (0u32..100).collect();
        let bubblesort = chair_prepare(RUNS, prepare, |mut data| bubblesort(&mut data));
        println!("bubblesort:\n{}", bubblesort.histogram::<50>());

        let std = chair_prepare(RUNS, prepare, |mut data| data.sort());
        println!("std:\n{}", std.histogram::<10>());

        let compare = agg_and_cmp![std, bubblesort];
        println!("{}", compare.table());
    }
}
