//! SIMD-accelerated search pipeline.
//!
//! Scores queries against quantized database vectors using nibble-split
//! lookup tables with architecture-specific SIMD kernels:
//! - NEON on ARM (sequential code layout)
//! - AVX2 on x86 (FAISS-style perm0-interleaved layout)

use rayon::prelude::*;
use crate::{BLOCK, FLUSH_EVERY};

// TODO: Port the full SIMD kernel from py-turboquant/src/lib.rs
// For now, provide a scalar fallback that produces correct results.

/// Per-query nibble LUTs for scoring.
struct QueryLut {
    uint8_luts: Vec<u8>,
    scale: f32,
    bias: f32,
}

/// Full search: rotation + LUT build + scoring + heap top-k.
/// Returns (scores_flat, indices_flat) each of length nq * k.
pub fn search(
    queries: &[f32],    // (nq, dim) row-major
    nq: usize,
    rotation: &[f32],   // (dim, dim) row-major
    blocked_codes: &[u8],
    centroids: &[f32],
    norms: &[f32],
    bits: usize,
    dim: usize,
    n_vectors: usize,
    n_blocks: usize,
    k: usize,
) -> (Vec<f32>, Vec<i64>) {
    let codes_per_byte = 8 / bits;
    let n_byte_groups = dim / codes_per_byte;
    let k = k.min(n_vectors);

    // Rotation: q_rot = queries @ rotation.T
    let mut q_rot = vec![0.0f32; nq * dim];
    for qi in 0..nq {
        for j in 0..dim {
            let mut sum = 0.0f32;
            for i in 0..dim {
                sum += queries[qi * dim + i] * rotation[j * dim + i];
            }
            q_rot[qi * dim + j] = sum;
        }
    }

    // Build LUTs
    let max_lut = {
        #[cfg(target_arch = "x86_64")]
        { (65535.0 / (n_byte_groups as f64 * 2.0)).floor().min(127.0) as f32 }
        #[cfg(not(target_arch = "x86_64"))]
        { 127.0f32 }
    };

    let query_luts: Vec<QueryLut> = (0..nq)
        .into_par_iter()
        .map(|qi| build_lut(&q_rot[qi * dim..(qi + 1) * dim], centroids, bits, dim, max_lut))
        .collect();

    // Scoring: scalar fallback (SIMD kernels to be ported)
    let mut all_scores = vec![0.0f32; nq * k];
    let mut all_indices = vec![0i64; nq * k];

    for qi in 0..nq {
        let lut = &query_luts[qi];
        let mut heap_s = vec![f32::NEG_INFINITY; k];
        let mut heap_i = vec![0u32; k];
        let mut heap_sz = 0usize;
        let mut heap_min = f32::NEG_INFINITY;
        let mut heap_mi = 0usize;

        for b in 0..n_blocks {
            let base_vec = b * BLOCK;
            let block_offset = b * n_byte_groups * BLOCK;

            for lane in 0..BLOCK {
                let vi = base_vec + lane;
                if vi >= n_vectors { break; }

                let mut score = 0.0f32;
                for g in 0..n_byte_groups {
                    let byte_val = blocked_codes[block_offset + g * BLOCK + lane] as usize;
                    let hi = byte_val >> 4;
                    let lo = byte_val & 0x0F;
                    let lut_hi_val = lut.scale * lut.uint8_luts[g * 32 + hi] as f32 + lut.bias;
                    let lut_lo_val = lut.scale * lut.uint8_luts[g * 32 + 16 + lo] as f32 + lut.bias;
                    score += lut_hi_val + lut_lo_val;
                }
                score *= norms[vi];

                if heap_sz < k {
                    heap_s[heap_sz] = score;
                    heap_i[heap_sz] = vi as u32;
                    heap_sz += 1;
                    if heap_sz == k {
                        heap_min = heap_s[0]; heap_mi = 0;
                        for h in 1..k {
                            if heap_s[h] < heap_min { heap_min = heap_s[h]; heap_mi = h; }
                        }
                    }
                } else if score > heap_min {
                    heap_s[heap_mi] = score;
                    heap_i[heap_mi] = vi as u32;
                    heap_min = heap_s[0]; heap_mi = 0;
                    for h in 1..k {
                        if heap_s[h] < heap_min { heap_min = heap_s[h]; heap_mi = h; }
                    }
                }
            }
        }

        // Sort heap
        let mut pairs: Vec<(f32, u32)> = heap_s[..heap_sz].iter()
            .zip(heap_i[..heap_sz].iter())
            .map(|(&s, &i)| (s, i)).collect();
        pairs.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        for j in 0..pairs.len().min(k) {
            all_scores[qi * k + j] = pairs[j].0;
            all_indices[qi * k + j] = pairs[j].1 as i64;
        }
    }

    (all_scores, all_indices)
}

fn build_lut(q_rot_row: &[f32], centroids: &[f32], bits: usize, dim: usize, max_lut: f32) -> QueryLut {
    let codes_per_byte = 8 / bits;
    let codes_per_nibble = codes_per_byte / 2;
    let n_byte_groups = dim / codes_per_byte;
    let code_mask = (1u16 << bits) - 1;

    let mut uint8_luts = vec![0u8; n_byte_groups * 32];
    let mut float_vals = vec![0.0f32; n_byte_groups * 32];
    let mut global_min = f32::MAX;
    let mut global_max = f32::MIN;

    for g in 0..n_byte_groups {
        let dim_start = g * codes_per_byte;

        for nibble_val in 0u16..16 {
            let mut s = 0.0f32;
            for c in 0..codes_per_nibble {
                let shift = (codes_per_nibble - 1 - c) * bits;
                let code = (nibble_val >> shift) & code_mask;
                s += q_rot_row[dim_start + c] * centroids[code as usize];
            }
            float_vals[g * 32 + nibble_val as usize] = s;
            global_min = global_min.min(s);
            global_max = global_max.max(s);
        }

        for nibble_val in 0u16..16 {
            let mut s = 0.0f32;
            for c in 0..codes_per_nibble {
                let shift = (codes_per_nibble - 1 - c) * bits;
                let code = (nibble_val >> shift) & code_mask;
                s += q_rot_row[dim_start + codes_per_nibble + c] * centroids[code as usize];
            }
            float_vals[g * 32 + 16 + nibble_val as usize] = s;
            global_min = global_min.min(s);
            global_max = global_max.max(s);
        }
    }

    let range = global_max - global_min;
    let scale = if range > 1e-10 { range / max_lut } else { 1.0 };
    let bias = global_min;

    for i in 0..float_vals.len() {
        uint8_luts[i] = ((float_vals[i] - bias) / scale).round().min(max_lut) as u8;
    }

    QueryLut { uint8_luts, scale, bias }
}
