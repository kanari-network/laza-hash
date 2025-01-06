use rand::{Rng, thread_rng};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::hash::Hasher;
use std::num::Wrapping;
use zeroize::{Zeroize, ZeroizeOnDrop};

const BLOCK_SIZE: usize = 128;
const ROUNDS: usize = 12;

const LAZA_IV: [u32; 32] = [
    
    0x61707865, 0x3320646E, 0x79622D32, 0x6B206574,
    
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    
    0x9E3779B9, 0x243F6A88, 0xB7E15162, 0x71374491,
    
    0xF1234567, 0xE89ABCDF, 0xD6789ABC, 0xC4567DEF,
    
    0x7FFFFFFF, 0x1FFFFFFF, 0x0FFFFFFF, 0x07FFFFFF,
    
    0xFFFFFFFF, 0xFFFFFFFE, 0xFFFFFFFD, 0xFFFFFFFC,
    
    0xD6D0E7F7, 0xA5D39983, 0x8C6F5171, 0x4A46D1B0
];

const STATE_SIZE: usize = 32;  // Increased from 16
const SALT_SIZE: usize = 32;
const KEY_SIZE: usize = 16;    // Increased from 8


#[derive(Clone, Zeroize, ZeroizeOnDrop)]
pub struct LazaHasher {
    /// Internal state array for hash computation
    state: [u32; STATE_SIZE],

    /// Input message buffer
    buffer: Vec<u8>,

    /// Number of blocks processed
    counter: u64,

    /// Random salt for hash uniqueness
    salt: [u8; SALT_SIZE],

    /// Number of mixing rounds
    rounds: usize,

    /// Optional key for MAC mode
    key: [u32; KEY_SIZE],

    /// Indicates if running in keyed mode
    is_keyed: bool,
}

const DOMAIN_HASH: u32 = 0x01;
const DOMAIN_KEYED: u32 = 0x02;

impl LazaHasher {
    fn compress(&mut self) {
        let mut working_state = self.state;

        // Enhanced domain separation with better mixing
        working_state[0] ^= if self.is_keyed {
            DOMAIN_KEYED.wrapping_mul(0x6A09E667)
        } else {
            DOMAIN_HASH.wrapping_mul(0xBB67AE85)
        };

        // Mix message length with counter
        working_state[12] ^= (self.counter & 0xFFFFFFFF) as u32;
        working_state[13] ^= (self.counter >> 32) as u32;
        working_state[14] ^= (self.buffer.len() as u32).rotate_left(16);

        // Process message with constant-time operations
        for i in 0..16 {
            let word = if i * 4 + 3 < self.buffer.len() {
                u32::from_le_bytes(self.buffer[i * 4..i * 4 + 4].try_into().unwrap())
            } else {
                0u32
            };
            working_state[i] = working_state[i].wrapping_add(word);
        }

        // Mix in salt
        for i in 0..8 {
            working_state[i] ^= u32::from_le_bytes(self.salt[i * 4..i * 4 + 4].try_into().unwrap());
        }

        // Improved key mixing for MAC mode
        if self.is_keyed {
            for i in 0..8 {
                working_state[i] = working_state[i]
                    .wrapping_add(self.key[i])
                    .rotate_right(11)
                    .wrapping_mul(0x9e3779b9);
            }
        }

        // Enhanced round function with round constants
        for r in 0..self.rounds {
            working_state[0] ^= r as u32; // Round constant

            // Column mixing with additional rotations
            for i in 0..4 {
                working_state[i] = working_state[i]
                    .wrapping_add(working_state[i + 4])
                    .rotate_right(7);
                working_state[i + 8] ^= working_state[i];
                working_state[i + 12] = working_state[i + 12]
                    .wrapping_add(working_state[i + 8])
                    .rotate_right(8);
            }

            // Diagonal mixing with stronger permutation
            for i in 0..4 {
                working_state[i] = working_state[i]
                    .wrapping_add(working_state[((i + 1) % 4) + 8])
                    .rotate_right(12);
                working_state[i + 4] ^= working_state[i];
                working_state[i + 8] = working_state[i + 8]
                    .wrapping_add(working_state[i + 4])
                    .rotate_right(16);
            }
        }

        // Stronger state update with additional mixing
        for i in 0..16 {
            self.state[i] = self.state[i]
                .wrapping_add(working_state[i])
                .rotate_right(i as u32 & 0x1F);
        }

        self.buffer.clear();
        self.counter = self.counter.wrapping_add(1);

        // Secure cleanup
        working_state.zeroize();
    }

    pub fn new() -> Self {
        let mut salt = [0u8; SALT_SIZE];
        thread_rng().fill(&mut salt);
        Self {
            state: LAZA_IV,  // Now matches 32-word size
            buffer: Vec::with_capacity(BLOCK_SIZE),
            counter: 0,
            salt,
            rounds: ROUNDS,
            key: [0u32; KEY_SIZE],
            is_keyed: false,
        }
    }

    pub fn with_key(key: &[u8; 32]) -> Self {
        let mut hasher = Self::new();
        for i in 0..8 {
            hasher.key[i] = u32::from_le_bytes(key[i * 4..(i + 1) * 4].try_into().unwrap());
        }
        hasher.is_keyed = true;
        hasher
    }

    fn finalize(&mut self) -> u64 {
        if !self.buffer.is_empty() {
            let orig_len = self.buffer.len();
            self.buffer.resize(BLOCK_SIZE, 0);
            self.buffer[orig_len] = 0x80;

            // Add length encoding
            let total_bits = (self.counter * BLOCK_SIZE as u64 + orig_len as u64) * 8;
            self.buffer[BLOCK_SIZE - 16..BLOCK_SIZE - 8].copy_from_slice(&total_bits.to_le_bytes());

            self.compress();
        }

        // Additional finalization mixing
        self.state[14] ^= (self.counter << 3) as u32;
        self.state[15] ^= if self.is_keyed {
            0x6a09e667
        } else {
            0xbb67ae85
        };

        // Output masking
        let mask = 0x243f6a88_85a308d3u64;
        ((self.state[0] as u64) << 32 | (self.state[1] as u64)) ^ mask
    }
}
// Add constant-time comparison
impl PartialEq for LazaHasher {
    fn eq(&self, other: &Self) -> bool {
        let mut result = 0u32;
        for (a, b) in self.state.iter().zip(other.state.iter()) {
            result |= a ^ b;
        }
        result == 0
    }
}

impl Hasher for LazaHasher {
    fn finish(&self) -> u64 {
        let mut hasher = self.clone();
        hasher.finalize()
    }

    fn write(&mut self, bytes: &[u8]) {
        debug_assert!(self.buffer.len() < BLOCK_SIZE);
        self.buffer.extend_from_slice(bytes);
        while self.buffer.len() >= BLOCK_SIZE {
            self.compress();
        }
    }
}

pub fn add_vectors_simd(a: &[f32], b: &[f32]) -> Vec<f32> {
    if is_x86_feature_detected!("avx2") {
        unsafe {
            // Implementation using AVX2 intrinsics
            add_vectors_avx2(a, b)
        }
    } else {
        // Fallback implementation
        a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect()
    }
}

#[target_feature(enable = "avx2")]
unsafe fn add_vectors_avx2(a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut result = Vec::with_capacity(a.len());

    for (a_chunk, b_chunk) in a.chunks(8).zip(b.chunks(8)) {
        unsafe {
            let a_vec = _mm256_loadu_ps(a_chunk.as_ptr());
            let b_vec = _mm256_loadu_ps(b_chunk.as_ptr());
            let sum = _mm256_add_ps(a_vec, b_vec);

            let mut temp = [0.0f32; 8];
            _mm256_storeu_ps(temp.as_mut_ptr(), sum);
            result.extend_from_slice(&temp[..a_chunk.len()]);
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use blake3::Hasher as Blake3Hasher;
    use criterion::black_box;
    #[cfg(feature = "parallel")]
    use rayon::prelude::*;
    use sha2::{Digest, Sha256};
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
        let sizes = [1, 16, 64, 256, 1024];
        let iterations = 25_000;
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
