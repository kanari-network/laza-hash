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
laza = "0.2.3"
```

## Usage

```rust
use laza::LazaHasher;
use std::hash::Hasher;

fn main() {
    // Test 1: Basic string with salt
    let mut hasher = LazaHasher::new_with_salt(1234);
    hasher.write(b"Test data 1");
    let hash1 = hasher.finish();
    println!("\n=== Test 1: Basic string ===");
    println!("Input: Test data 1");
    println!("Salt: 1234");
    println!("Hash: {:016x}", hash1);

    // Test 2: Same string, different salt
    let mut hasher = LazaHasher::new_with_salt(5678);
    hasher.write(b"Test data 1");
    let hash2 = hasher.finish();
    println!("\n=== Test 2: Same string, different salt ===");
    println!("Input: Test data 1");
    println!("Salt: 5678"); 
    println!("Hash: {:016x}", hash2);

    // Test 3: Different string, same salt
    let text = "Different test data";
    let mut hasher = LazaHasher::new_with_salt(5678);
    hasher.write(text.as_bytes());
    let hash3 = hasher.finish();
    println!("\n=== Test 3: Different string ===");
    println!("Input: {}", text);
    println!("Salt: 5678");
    println!("Hash: {:016x}", hash3);
}
```

## Parallel Processing
The library automatically uses available system threads for parallel processing. The number of threads is determined by:

- Default: Available system parallelism
- Falls back to single thread if parallelism info cannot be obtained