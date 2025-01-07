#[cfg(test)]
mod tests {
    use crate::LazaHasher;

    // use super::*;
    use blake3::Hasher as Blake3Hasher;
    use criterion::black_box;
    #[cfg(feature = "parallel")]
    use rayon::prelude::*;
    use sha2::{Digest, Sha256};
    use std::hash::Hasher;
    use std::sync::Arc;
    use std::thread;
    use std::time::Instant;

    type HashFn = fn(&[u8]) -> Vec<u8>;

    #[test]
    fn benchmark_hashers() {
        let thread_count = thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1);
        let sizes = [1, 16, 64, 256, 1024];
        let iterations = 10000 * thread_count;
        let batch_size = 100;

        // Define static hash functions
        let laza_hash: HashFn = |data| {
            let mut hasher = LazaHasher::new();
            hasher.write(data);
            hasher.finish().to_le_bytes().to_vec()
        };

        let sha256_hash: HashFn = |data| {
            let mut hasher = Sha256::new();
            hasher.update(data);
            hasher.finalize().to_vec()
        };

        let blake3_hash: HashFn = |data| {
            Blake3Hasher::new()
                .update(data)
                .finalize()
                .as_bytes()
                .to_vec()
        };

        println!(
            "\nOptimized Performance Benchmark (threads: {})",
            thread_count
        );
        println!("==============================================");
        println!("Data Size | LAZA (MB/s) | SHA256 (MB/s) | BLAKE3 (MB/s) | vs SHA256 | vs BLAKE3");
        println!("---------+-------------+--------------+--------------+----------+---------");

        for &size_kb in &sizes {
            let size = size_kb * 1024;
            let data = Arc::new(vec![0x5au8; size]);

            let bench = |f: HashFn| -> f64 {
                let data = Arc::clone(&data);
                let start = Instant::now();
                (0..iterations / batch_size).into_par_iter().for_each(|_| {
                    for _ in 0..batch_size {
                        black_box(f(&data));
                    }
                });
                let time = start.elapsed();
                (size as f64 * iterations as f64) / (1024.0 * 1024.0 * time.as_secs_f64())
            };

            let laza = bench(laza_hash);
            let sha256 = bench(sha256_hash);
            let blake3 = bench(blake3_hash);

            println!(
                "{:6} KB | {:9.2} | {:10.2} | {:10.2} | {:8.2}x | {:7.2}x",
                size_kb,
                laza,
                sha256,
                blake3,
                laza / sha256,
                laza / blake3
            );
        }
    }

    #[test]
    fn benchmark_laza() {
        // Thread pool setup
        let threads = thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1);
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .unwrap();

        // Test parameters
        let sizes = [1, 16, 64, 256, 1024, 102400];
        let iterations = 10_000;
        let runs = 5;
        let batch_size = 100;

        // Warmup phase
        let warmup_data = Arc::new(vec![0x5au8; 64 * 1024]);
        for _ in 0..1000 {
            let mut hasher = LazaHasher::new();
            hasher.write(&warmup_data);
            black_box(hasher.finish());
        }

        println!("\nLAZA Performance Benchmark (threads: {})", threads);
        println!("========================================");
        println!("Data Size | Throughput (GB/s) | StdDev");
        println!("---------+------------------+--------");

        for &size_kb in &sizes {
            let size = size_kb * 1024;
            let data = Arc::new(vec![0x5au8; size]);
            let mut results = Vec::with_capacity(runs);

            for _ in 0..runs {
                let start = Instant::now();
                (0..iterations / batch_size).into_par_iter().for_each(|_| {
                    let data = Arc::clone(&data);
                    for _ in 0..batch_size {
                        let mut hasher = LazaHasher::new();
                        hasher.write(&data);
                        black_box(hasher.finish());
                    }
                });
                let elapsed = start.elapsed();
                let throughput = (size as f64 * iterations as f64)
                    / (1024.0 * 1024.0 * 1024.0 * elapsed.as_secs_f64());
                results.push(throughput);
            }

            results.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let median = results[runs / 2];
            let stddev =
                (results.iter().map(|x| (x - median).powi(2)).sum::<f64>() / runs as f64).sqrt();

            println!("{:6} KB | {:14.2} | {:.3}", size_kb, median, stddev);
        }
    }

}
