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
laza = "0.1.2"
```

## Usage

```rust
use laza::LazaHasher;

// Create a new hasher
let mut hasher = LazaHasher::new();

// Add data to be hashed
hasher.write(b"Hello world");

// Get the hash result
let hash = hasher.finish();
```

## Parallel Processing
The library automatically uses available system threads for parallel processing. The number of threads is determined by:

- Default: Available system parallelism
- Falls back to single thread if parallelism info cannot be obtained