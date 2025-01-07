use rand::{Rng, thread_rng};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::hash::Hasher;
use zeroize::{Zeroize, ZeroizeOnDrop};

mod test;

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

const SIGMA: [[usize; 16]; 10] = [
    [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 ],
    [ 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 ],
    [ 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 ],
    [ 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 ],
    [ 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13 ],
    [ 2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9 ],
    [ 12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11 ],
    [ 13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10 ],
    [ 6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5 ],
    [ 10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0 ],
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

impl LazaHasher {
    #[inline(always)]
    fn g(&mut self, a: usize, b: usize, c: usize, d: usize, x: u32, y: u32) {
        self.state[a] = self.state[a].wrapping_add(self.state[b]).wrapping_add(x);
        self.state[d] = (self.state[d] ^ self.state[a]).rotate_right(16);
        self.state[c] = self.state[c].wrapping_add(self.state[d]);
        self.state[b] = (self.state[b] ^ self.state[c]).rotate_right(12);
        self.state[a] = self.state[a].wrapping_add(self.state[b]).wrapping_add(y);
        self.state[d] = (self.state[d] ^ self.state[a]).rotate_right(8);
        self.state[c] = self.state[c].wrapping_add(self.state[d]);
        self.state[b] = (self.state[b] ^ self.state[c]).rotate_right(7);
    }

    fn compress(&mut self) {
        let mut v = [0u32; 16];
        v[..8].copy_from_slice(&self.state[..8]);
        v[8..].copy_from_slice(&LAZA_IV[..8]);

        let mut m = [0u32; 16];
        for i in 0..16 {
            m[i] = u32::from_le_bytes(self.buffer[i*4..(i+1)*4].try_into().unwrap());
        }

        for i in 0..self.rounds {
            let s = &SIGMA[i % 10];
            // Column steps
            self.g(0, 4, 8, 12, m[s[0]], m[s[1]]);
            self.g(1, 5, 9, 13, m[s[2]], m[s[3]]);
            self.g(2, 6, 10, 14, m[s[4]], m[s[5]]);
            self.g(3, 7, 11, 15, m[s[6]], m[s[7]]);
            // Diagonal steps
            self.g(0, 5, 10, 15, m[s[8]], m[s[9]]);
            self.g(1, 6, 11, 12, m[s[10]], m[s[11]]);
            self.g(2, 7, 8, 13, m[s[12]], m[s[13]]);
            self.g(3, 4, 9, 14, m[s[14]], m[s[15]]);
        }

        for i in 0..8 {
            self.state[i] ^= v[i] ^ v[i + 8];
        }

        self.buffer.clear();
        self.counter = self.counter.wrapping_add(1);
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

