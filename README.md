# Chairmark

A benchmaking library in rust.

## quick start
```rust
use chairmark::{chair, agg_and_cmp};

// custom function for sorting data
fn bubblesort(data: &mut Vec<u32>) {
    /* your fancy sorting algorithm */
}

fn main() {
    const RUNS: usize = 1_000;
    let prepare = |_| (0u32..1_000).collect();
    let bubblesort = chair_prepare(RUNS, prepare, |mut data| bubblesort(&mut data));
    let std = chair_prepare(RUNS, prepare, |mut data| data.sort());

    let compare = agg_and_cmp![std, bubblesort];
    println!("{}", compare);
}
```

## TODOs
* [ ] add more examples in README