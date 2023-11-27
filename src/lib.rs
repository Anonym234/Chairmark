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

use std::{
    fmt::Display,
    time::{Duration, Instant},
};

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

mod multiset;
use multiset::MultiSet;

mod table {
    pub struct Table<N: AsRef<str>, T: AsRef<str>, const WIDTH: usize> {
        header: Option<Row<N, N, WIDTH>>,
        alignment: Alignments<WIDTH>,
        rows: Vec<Row<N, T, WIDTH>>,
    }

    struct Row<N: AsRef<str>, T: AsRef<str>, const WIDTH: usize> {
        header: N,
        data: [T; WIDTH],
    }

    impl<N: AsRef<str>, T: AsRef<str>, const WIDTH: usize> From<(N, [T; WIDTH])> for Row<N, T, WIDTH> {
        fn from(value: (N, [T; WIDTH])) -> Self {
            let (header, data) = value;
            Self { header, data }
        }
    }

    impl<N, const INIT: usize, const WIDTH: usize> From<[N; INIT]> for Row<N, N, WIDTH>
    where
        N: AsRef<str>,
    {
        fn from(value: [N; INIT]) -> Self {
            assert_eq!(WIDTH + 1, INIT);

            value.chunks()
            
            Self {
                header: 
            }
        }
    }

    enum Alignment {
        Left,
        Right,
        Center,
    }

    struct Alignments<const WIDTH: usize> {
        header: Alignment,
        data: [Alignment; WIDTH],
    }

    impl<const WIDTH: usize> Default for Alignments<WIDTH> {
        fn default() -> Self {
            Self {
                header: Alignment::Left,
                data: [Alignment::Right; WIDTH],
            }
        }
    }

    pub struct Builder<N: AsRef<str>, T: AsRef<str>, const WIDTH: usize> {
        header: Option<Row<N, T, WIDTH>>,
        alignment: Option<Alignments<WIDTH>>,
        rows: Vec<Row<N, T, WIDTH>>,
    }

    impl<N: AsRef<str>, T: AsRef<str>, const WIDTH: usize> Builder<N, T, WIDTH> {
        pub fn new() -> Self {
            Self {
                header: None,
                alignment: None,
                rows: Vec::new(),
            }
        }

        pub fn header(&mut self, header: impl Into<Row<N, N, WIDTH>>) {
            assert!(self.header.is_none());

            self.header = Some(header.into());
        }

        pub fn alignment(&mut self, alignment: impl Into<Alignments<WIDTH>>) {
            assert!(self.alignment.is_none());

            self.alignment = Some(alignment.into());
        }

        pub fn add_row(&mut self, row: impl Into<Row<N, T, WIDTH>>) {
            self.rows.push(row.into());
        }

        pub fn make(self) -> Table<N, T, WIDTH> {
            Table {
                header: self.header,
                alignment: self.alignment.unwrap_or_default(),
                rows: self.rows,
            }
        }
    }
}
use table::Table;

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
#[derive(Debug, Clone, Default)]
pub struct Measurements {
    durations: MultiSet<Duration>,
}

impl Measurements {
    /// new, empty measurements
    fn new() -> Self {
        Self::default()
    }

    /// accept a new data point (duration)
    fn accept(&mut self, duration: Duration) {
        self.durations.insert(duration);
    }

    /// get number of datapoints stored
    pub fn size(&self) -> usize {
        self.durations.size()
    }

    /// checks whether no datapoints are stored
    pub fn is_empty(&self) -> bool {
        self.durations.is_empty()
    }

    /// shortest duration of collected data
    ///
    /// # panic
    /// panis if `self.is_empty()`
    pub fn min(&self) -> Duration {
        *self.durations.min().unwrap()
    }

    /// longest duration of collected data
    ///
    /// # panic
    /// panis if `self.is_empty()`
    pub fn max(&self) -> Duration {
        *self.durations.max().unwrap()
    }

    /// mean duration of collected data (arithmetic mean)
    ///
    /// # panic
    /// panis if `self.is_empty()`
    pub fn arith_mean(&self) -> Duration {
        self.durations.iter().reduce(|l, r| l + r).unwrap() / self.size() as u32
    }

    /// median duration of collected data
    ///
    /// **NOTE**: right-biased in an even length data collection
    ///
    /// # panic
    /// panis if `self.is_empty()`
    pub fn median(&self) -> Duration {
        self.durations.iter().take(self.size() / 2).last().unwrap()
    }

    /// mode duration of collected data (the most often occured duration)
    ///
    /// **NOTE**: this might yield unreliable results
    ///
    /// # panic
    /// panis if `self.is_empty()`
    pub fn mode(&self) -> Duration {
        *self
            .durations
            .iter_counts()
            .reduce(|l @ (_, lc), r @ (_, rc)| if rc > lc { r } else { l })
            .unwrap()
            .0
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

    /// aggregate all other information into one struct
    ///
    /// # panics
    /// since it uses the other aggregate functions, if the collection is empty, a `panic!` is invoked
    pub fn aggregate<T>(&self) -> Aggregate<T>
    where
        Duration: Into<T>,
    {
        Aggregate {
            min: self.min().into(),
            max: self.max().into(),
            arith_mean: self.arith_mean().into(),
            median: self.median().into(),
            mode: self.mode().into(),
            variance: self.variance().into(),
        }
    }
}

impl<'a> IntoIterator for &'a Measurements {
    type Item = Duration;
    type IntoIter = <&'a MultiSet<Duration> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        (&self.durations).into_iter()
    }
}

impl IntoIterator for Measurements {
    type Item = Duration;
    type IntoIter = <MultiSet<Duration> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.durations.into_iter()
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
    mode: T,
    variance: T,
}

impl<T> Aggregate<T> {
    /// getter functions and names for all stored data
    const GET_DATA: [(&'static str, fn(&Self) -> &T); 6] = [
        ("min", Self::min),
        ("max", Self::max),
        ("arith_mean", Self::arith_mean),
        ("median", Self::median),
        ("mode", Self::mode),
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

    /// get mode
    pub fn mode(&self) -> &T {
        &self.mode
    }

    /// get variance
    pub fn variance(&self) -> &T {
        &self.variance
    }

    /// get all data with names
    ///
    /// returns an array with tuples (name, value)
    pub fn all(&self) -> [(&'static str, &T); 6] {
        [
            ("min", self.min()),
            ("max", self.max()),
            ("arith_mean", self.arith_mean()),
            ("median", self.median()),
            ("mode", self.mode()),
            ("variance", self.variance()),
        ]
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
            mode: value.mode.into(),
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
    fn table(&self) -> Table {
        todo!();
    }
}

/// runs a chairmark on a single function
///
/// It executes the function `runs` times, returning the measurements.
/// These [`Measurements`] can be aggregated etc..
///
/// # Example
/// ```rust
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
/// fn prepare(run_id: usize) -> Vec<u32> {
///     (0..run_id as u32).collect()
/// }
/// fn main() {
///     let agg = chair_prepare(1_000, prepare, |data| data.sort()).aggregate::<Time>();
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

    measurements
}
