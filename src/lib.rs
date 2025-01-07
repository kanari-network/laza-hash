use rand::{Rng, thread_rng};
use rayon::{iter::{IntoParallelRefIterator, ParallelIterator}, slice::ParallelSlice};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::hash::Hasher;
use zeroize::{Zeroize, ZeroizeOnDrop};

mod test;

const BLOCK_SIZE: usize = 128;
const ROUNDS: usize = 12;

const LAZA_IV: [u32; 32] = [
    0x61707865, 0x3320646E, 0x79622D32, 0x6B206574, 0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x9E3779B9, 0x243F6A88, 0xB7E15162, 0x71374491,
    0xF1234567, 0xE89ABCDF, 0xD6789ABC, 0xC4567DEF, 0x7FFFFFFF, 0x1FFFFFFF, 0x0FFFFFFF, 0x07FFFFFF,
    0xFFFFFFFF, 0xFFFFFFFE, 0xFFFFFFFD, 0xFFFFFFFC, 0xD6D0E7F7, 0xA5D39983, 0x8C6F5171, 0x4A46D1B0,
];

const SIGMA: [[usize; 16]; 10] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
    [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
    [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
    [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13],
    [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
    [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11],
    [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
    [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5],
    [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0],
];

const STATE_SIZE: usize = 32; // Increased from 16
const SALT_SIZE: usize = 32;
const KEY_SIZE: usize = 16; // Increased from 8

const CHUNK_SIZE: usize = 16384; // 16KB chunks for parallel processing
const CACHE_LINE: usize = 64;    // Common cache line size

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

#[cfg(target_arch = "x86_64")]
mod simd {
    use super::*;
    
    #[cfg(target_feature = "avx512f")]
    pub(crate) unsafe fn compress_avx512(state: &mut [u32; STATE_SIZE], chunks: &[&[u8]]) -> bool {
        // AVX-512 implementation
        unimplemented!()
    }

    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn compress_avx2(state: &mut [u32; STATE_SIZE], chunks: &[&[u8]]) -> bool {
        if chunks.is_empty() { return false; }

        unsafe {
            let mut acc0 = _mm256_loadu_si256(state[0..8].as_ptr() as *const __m256i);
            let mut acc1 = _mm256_loadu_si256(state[8..16].as_ptr() as *const __m256i);
            let mut acc2 = _mm256_loadu_si256(state[16..24].as_ptr() as *const __m256i);
            let mut acc3 = _mm256_loadu_si256(state[24..32].as_ptr() as *const __m256i);

            for chunk in chunks {
                let chunk_vec = _mm256_loadu_si256(chunk.as_ptr() as *const __m256i);
                acc0 = _mm256_add_epi32(acc0, chunk_vec);
                // Additional SIMD operations here
            }

            _mm256_storeu_si256(state[0..8].as_mut_ptr() as *mut __m256i, acc0);
            _mm256_storeu_si256(state[8..16].as_mut_ptr() as *mut __m256i, acc1);
            _mm256_storeu_si256(state[16..24].as_mut_ptr() as *mut __m256i, acc2);
            _mm256_storeu_si256(state[24..32].as_mut_ptr() as *mut __m256i, acc3);
        }
        true
    }
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::*;
    use std::arch::x86_64::*;

    /// Safety: Requires AVX2 support and properly aligned state/buffer
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn compress_avx2(state: &mut [u32; STATE_SIZE], buffer: &[u8]) -> bool {
        if buffer.len() < BLOCK_SIZE {
            return false;
        }

        // Convert buffer to blocks safely
        let mut msg_blocks = [[0u8; 32]; 4];
        for (i, chunk) in buffer.chunks(32).take(4).enumerate() {
            msg_blocks[i][..chunk.len()].copy_from_slice(chunk);
        }

        unsafe {
            // Load state vectors
            let working_vec0 = _mm256_loadu_si256(state[0..8].as_ptr() as *const __m256i);
            let working_vec1 = _mm256_loadu_si256(state[8..16].as_ptr() as *const __m256i);
            let working_vec2 = _mm256_loadu_si256(state[16..24].as_ptr() as *const __m256i);
            let working_vec3 = _mm256_loadu_si256(state[24..32].as_ptr() as *const __m256i);

            let mut result_vec0 = working_vec0;
            
            // Process blocks
            for block in &msg_blocks {
                let msg_vec = _mm256_loadu_si256(block.as_ptr() as *const __m256i);
                result_vec0 = _mm256_add_epi32(result_vec0, msg_vec);
            }

            // Store results
            _mm256_storeu_si256(state[0..8].as_mut_ptr() as *mut __m256i, result_vec0);
            _mm256_storeu_si256(state[8..16].as_mut_ptr() as *mut __m256i, working_vec1);
            _mm256_storeu_si256(state[16..24].as_mut_ptr() as *mut __m256i, working_vec2);
            _mm256_storeu_si256(state[24..32].as_mut_ptr() as *mut __m256i, working_vec3);
        }

        true
    }
}

impl LazaHasher {
    pub fn new_with_salt(salt: u64) -> Self {
        let mut hasher = Self::new();
        
        // Mix salt into initial state
        let salt_bytes = salt.to_le_bytes();
        for i in 0..8 {
            hasher.state[i] ^= u32::from_le_bytes([
                salt_bytes[i % 8],
                salt_bytes[(i + 1) % 8],
                salt_bytes[(i + 2) % 8],
                salt_bytes[(i + 3) % 8]
            ]);
        }

        // Additional state mixing
        for i in 0..STATE_SIZE {
            hasher.state[i] = hasher.state[i].wrapping_mul(0x6a09e667);
            hasher.state[i] = hasher.state[i].rotate_right(i as u32 + 1);
        }
        
        hasher
    }

    pub fn write(&mut self, bytes: &[u8]) {
        if bytes.len() > CHUNK_SIZE {
            let chunks: Vec<_> = bytes.par_chunks(CHUNK_SIZE).collect();
            let mut states: Vec<[u32; STATE_SIZE]> = chunks
                .par_iter()
                .map(|chunk| {
                    let mut state = self.state;
                    unsafe {
                        if is_x86_feature_detected!("avx512f") && cfg!(target_feature = "avx512f") {
                            simd::compress_avx2(&mut state, &[chunk]);
                        } else if is_x86_feature_detected!("avx2") {
                            simd::compress_avx2(&mut state, &[chunk]); 
                        }
                    }
                    #[cfg(target_arch = "x86_64")]
                    state
                })
                .collect();

            // Combine chunk results
            self.state = states.into_iter().fold(self.state, |acc, state| {
                let mut result = acc;
                for (a, b) in result.iter_mut().zip(state.iter()) {
                    *a = a.wrapping_add(*b);
                }
                result
            });
        } else {
            // Original processing for small inputs
            // ...existing code...
        }
    }


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
        if cfg!(target_arch = "x86_64") && is_x86_feature_detected!("avx2") {
            unsafe {
                if !avx2::compress_avx2(&mut self.state, &self.buffer) {
                    self.compress_scalar();
                    return;
                }
            }
        } else {
            self.compress_scalar();
        }
        
        self.buffer.clear();
        self.counter = self.counter.wrapping_add(1);
    }

    #[inline(always)]
    fn compress_scalar(&mut self) {
        let mut m = [0u32; 16];
        
        // Vectorized buffer loading
        let mut ptr = self.buffer.as_ptr();
        unsafe {
            for chunk in m.chunks_exact_mut(4) {
                let chunk_ptr = chunk.as_mut_ptr();
                std::ptr::copy_nonoverlapping(ptr, chunk_ptr as *mut u8, 16);
                ptr = ptr.add(16);
            }
        }

        // Unrolled rounds
        for i in 0..self.rounds {
            let s = &SIGMA[i % 10];
            
            // Unrolled quarter rounds
            for j in 0..4 {
                let base = j * 4;
                self.g(base, base+1, base+2, base+3, 
                      m[s[j*2]], m[s[j*2+1]]);
            }
        }

        // Optimized feed-forward
        for i in (0..8).step_by(4) {
            self.state[i] ^= self.state[i+8];
            self.state[i+1] ^= self.state[i+9];
            self.state[i+2] ^= self.state[i+10];
            self.state[i+3] ^= self.state[i+11];
        }
    }

    pub fn new() -> Self {
        let mut salt = [0u8; SALT_SIZE];
        thread_rng().fill(&mut salt);
        Self {
            state: LAZA_IV, // Now matches 32-word size
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
            self.compress();
        }

        // Enhanced finalization mixing
        for i in 0..4 {
            let t = self.state[i].wrapping_add(self.state[i + 4]);
            self.state[i + 8] ^= t.rotate_left(i as u32 * 7 + 1);
        }

        // Avalanche
        let mut h = self.state[0].wrapping_mul(0x85ebca6b);
        h ^= h >> 13;
        h = h.wrapping_mul(0xc2b2ae35);
        h ^= h >> 16;
        
        ((h as u64) << 32) | (self.state[1] as u64)
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

#[derive(Debug, Clone)]
pub struct LazaHash {
    state: [u32; STATE_SIZE],
    buffer: [u8; BLOCK_SIZE],
    salt: [u32; SALT_SIZE / 4],
    count: u64,
    buflen: usize,
}

impl LazaHash {
    fn quarter_round(state: &mut [u32], a: usize, b: usize, c: usize, d: usize) {
        // Create a temporary array to store values
        let mut tmp = [state[a], state[b], state[c], state[d]];

        // Perform operations on the temporary array
        tmp[0] = tmp[0].wrapping_add(tmp[1]);
        tmp[3] ^= tmp[0];
        tmp[3] = tmp[3].rotate_right(16);

        tmp[2] = tmp[2].wrapping_add(tmp[3]);
        tmp[1] ^= tmp[2];
        tmp[1] = tmp[1].rotate_right(12);

        tmp[0] = tmp[0].wrapping_add(tmp[1]);
        tmp[3] ^= tmp[0];
        tmp[3] = tmp[3].rotate_right(8);

        tmp[2] = tmp[2].wrapping_add(tmp[3]);
        tmp[1] ^= tmp[2];
        tmp[1] = tmp[1].rotate_right(7);

        // Write back results
        state[a] = tmp[0];
        state[b] = tmp[1];
        state[c] = tmp[2];
        state[d] = tmp[3];
    }

    fn compress(&mut self) {
        let mut working_state = self.state;
    
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") {
            unsafe {
                for round in 0..ROUNDS {
                    // Process 4 quarter rounds at once using AVX2
                    for i in 0..4 {
                        let idx = i * 4;
                        let values = _mm256_setr_epi32(
                            working_state[idx] as i32,
                            working_state[idx + 1] as i32, 
                            working_state[idx + 2] as i32,
                            working_state[idx + 3] as i32,
                            0, 0, 0, 0
                        );
    
                        // Vectorized quarter round
                        let added = _mm256_add_epi32(values, _mm256_srli_epi32(values, 16));
                        let mixed = _mm256_xor_si256(added, _mm256_srli_epi32(values, 12));
                        let rotated = _mm256_or_si256(
                            _mm256_slli_epi32(mixed, 16),
                            _mm256_srli_epi32(mixed, 16)
                        );
    
                        // Store results
                        let mut result = [0i32; 8];
                        _mm256_storeu_si256(result.as_mut_ptr() as *mut __m256i, rotated);
                        for j in 0..4 {
                            working_state[idx + j] = result[j] as u32;
                        }
                    }
    
                    // Diagonal rounds using permutation
                    let perm = SIGMA[round % 10];
                    for i in 0..4 {
                        let idx = i * 4;
                        Self::quarter_round(
                            &mut working_state,
                            perm[idx],
                            perm[idx + 1],
                            perm[idx + 2],
                            perm[idx + 3]
                        );
                    }
                }
            }
        } else {
            // Fallback to scalar implementation
            for round in 0..ROUNDS {
                for i in 0..4 {
                    let idx = i * 4;
                    Self::quarter_round(&mut working_state, idx, idx + 1, idx + 2, idx + 3);
                }
    
                let perm = SIGMA[round % 10];
                for i in 0..4 {
                    let idx = i * 4;
                    Self::quarter_round(
                        &mut working_state,
                        perm[idx],
                        perm[idx + 1],
                        perm[idx + 2],
                        perm[idx + 3]
                    );
                }
            }
        }
    
        // Feed-forward with bounds check
        debug_assert!(working_state.len() >= STATE_SIZE);
        for i in 0..STATE_SIZE {
            self.state[i] = self.state[i].wrapping_add(working_state[i]);
        }
    }
}

impl Default for LazaHash {
    fn default() -> Self {
        let mut hasher = LazaHash {
            state: [0u32; STATE_SIZE],
            buffer: [0u8; BLOCK_SIZE],
            salt: [0u32; SALT_SIZE / 4],
            count: 0,
            buflen: 0,
        };

        // Initialize state with IV
        hasher.state.copy_from_slice(&LAZA_IV);

        // Generate random salt
        let mut rng = thread_rng();
        for s in hasher.salt.iter_mut() {
            *s = rng.r#gen();
        }

        hasher
    }
}

impl Drop for LazaHash {
    fn drop(&mut self) {
        self.state.zeroize();
        self.buffer.zeroize();
        self.salt.zeroize();
    }
}

impl Hasher for LazaHash {
    fn finish(&self) -> u64 {
        // Combine last two state words for output
        ((self.state[0] as u64) << 32) | (self.state[1] as u64)
    }

    fn write(&mut self, input: &[u8]) {
        for byte in input {
            self.buffer[self.buflen] = *byte;
            self.buflen += 1;

            if self.buflen == BLOCK_SIZE {
                self.compress();
                self.count += BLOCK_SIZE as u64;
                self.buflen = 0;
            }
        }
    }
}

impl ZeroizeOnDrop for LazaHash {}