# LAZA

A high-performance hashing library with parallel processing capabilities.

## Features
- Fast parallel hashing
- Comparison benchmarks against SHA-256 and BLAKE3
- Configurable thread pool settings

## Installation

Add this to your `Cargo.toml`:

```toml 
[dependencies]
laza = "0.1.6"
```

## Usage

```rust
use laza::LazaHasher;
use std::hash::Hasher;

fn main() {
    // Basic usage
    let mut hasher = LazaHasher::new();
    hasher.write(b"Hello, World!");
    let hash = hasher.finish();
    println!("Hash: {:x}", hash);

    // Hash a string
    let text = "Sample text";
    let mut hasher = LazaHasher::new();
    hasher.write(text.as_bytes());
    let hash = hasher.finish();
    println!("String hash: {:x}", hash);

    // Hash a file
    let data = std::fs::read("example.txt").unwrap();
    let mut hasher = LazaHasher::new();
    hasher.write(&data);
    let hash = hasher.finish();
    println!("File hash: {:x}", hash);
}
```

## Parallel Processing
The library automatically uses available system threads for parallel processing. The number of threads is determined by:

- Default: Available system parallelism
- Falls back to single thread if parallelism info cannot be obtained