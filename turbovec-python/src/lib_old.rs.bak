use numpy::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

const BLOCK: usize = 32;

/// SIMD dot product of two f32 slices (must be same length, multiple of 4).
#[inline]
fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    debug_assert_eq!(n, b.len());

    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);
        let ap = a.as_ptr();
        let bp = b.as_ptr();
        let mut i = 0;
        while i + 15 < n {
            acc0 = vfmaq_f32(acc0, vld1q_f32(ap.add(i)), vld1q_f32(bp.add(i)));
            acc1 = vfmaq_f32(acc1, vld1q_f32(ap.add(i + 4)), vld1q_f32(bp.add(i + 4)));
            acc2 = vfmaq_f32(acc2, vld1q_f32(ap.add(i + 8)), vld1q_f32(bp.add(i + 8)));
            acc3 = vfmaq_f32(acc3, vld1q_f32(ap.add(i + 12)), vld1q_f32(bp.add(i + 12)));
            i += 16;
        }
        acc0 = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
        while i + 3 < n {
            acc0 = vfmaq_f32(acc0, vld1q_f32(ap.add(i)), vld1q_f32(bp.add(i)));
            i += 4;
        }
        let mut sum = vaddvq_f32(acc0);
        while i < n {
            sum += *ap.add(i) * *bp.add(i);
            i += 1;
        }
        sum
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut sum = 0.0f32;
        for i in 0..n {
            sum += a[i] * b[i];
        }
        sum
    }
}

// Max byte groups per uint16 flush batch.
// With 7-bit LUT entries, hi+lo max = 254 per group.
// 256 * 254 = 65,024 < 65,535 (uint16 max).
const FLUSH_EVERY: usize = 256;

// Queries processed per rayon work unit.
// Larger = less scheduling overhead, but need enough units to fill all cores.
// 16 queries × 24KB LUT = 384KB (fits in L2).
const QUERY_BATCH: usize = 16;

/// Pack bit-plane codes into SIMD-blocked layout.
/// On x86: FAISS-style perm0-interleaved layout for AVX2 cross-lane compatibility.
/// On ARM: sequential layout (no interleaving needed, NEON has no cross-lane issue).
fn repack_inner(
    packed_codes: ArrayView2<u8>,
    bits: usize,
    dim: usize,
) -> (Array1<u8>, usize) {
    let n = packed_codes.nrows();
    let bytes_per_plane = dim / 8;
    let codes_per_byte = 8 / bits;
    let n_byte_groups = dim / codes_per_byte;
    let n_blocks = (n + BLOCK - 1) / BLOCK;
    let blocked_size = n_blocks * n_byte_groups * BLOCK;

    let perm0: [usize; 16] = [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15];

    // Step 1: Extract packed byte per vector per group (same as original repack_inner).
    // Each byte has codes packed in nibble-aligned positions:
    //   4-bit: hi nibble = code0, lo nibble = code1
    //   2-bit: bits 7-6 = code0, 5-4 = code1, 3-2 = code2, 1-0 = code3
    let mut codes_flat = vec![vec![0u8; n_byte_groups]; n];

    for vec_idx in 0..n {
        for g in 0..n_byte_groups {
            let dim_start = g * codes_per_byte;
            let mut byte_val = 0u8;
            for c in 0..codes_per_byte {
                let j = dim_start + c;
                let byte_in_plane = j / 8;
                let bit_in_byte = 7 - (j % 8);
                let mask = 1u8 << bit_in_byte;

                let mut code = 0u8;
                for p in 0..bits {
                    let plane_byte = packed_codes[[vec_idx, p * bytes_per_plane + byte_in_plane]];
                    if plane_byte & mask != 0 {
                        code |= 1 << p;
                    }
                }

                let shift = if bits == 3 {
                    (codes_per_byte - 1 - c) * 4
                } else {
                    (codes_per_byte - 1 - c) * bits
                };
                byte_val |= code << shift;
            }
            codes_flat[vec_idx][g] = byte_val;
        }
    }

    // Step 2: Pack into platform-specific SIMD layout
    (pack_blocked(n, n_blocks, n_byte_groups, blocked_size, &codes_flat, &perm0), n_blocks)
}

#[cfg(target_arch = "x86_64")]
fn pack_blocked(n: usize, n_blocks: usize, n_byte_groups: usize, blocked_size: usize,
                codes_flat: &[Vec<u8>], perm0: &[usize; 16]) -> Array1<u8> {
    // FAISS layout: split each byte into hi/lo nibbles, interleave with perm0.
    // bytes 0-15 = hi nibbles (sq0) for perm0-interleaved vector pairs
    // bytes 16-31 = lo nibbles (sq1) for same pairs
    let mut blocked = vec![0u8; blocked_size];
    for block_idx in 0..n_blocks {
        let base_vec = block_idx * BLOCK;
        for g in 0..n_byte_groups {
            let out_offset = (block_idx * n_byte_groups + g) * BLOCK;
            for j in 0..16 {
                let va = base_vec + perm0[j];
                let vb = base_vec + perm0[j] + 16;
                let ba = if va < n { codes_flat[va][g] } else { 0 };
                let bb = if vb < n { codes_flat[vb][g] } else { 0 };
                // hi nibbles of both vectors packed into one byte
                blocked[out_offset + j] = (ba >> 4) | ((bb >> 4) << 4);
                // lo nibbles of both vectors packed into one byte
                blocked[out_offset + 16 + j] = (ba & 0x0F) | ((bb & 0x0F) << 4);
            }
        }
    }
    Array1::from_vec(blocked)
}

#[cfg(not(target_arch = "x86_64"))]
fn pack_blocked(n: usize, n_blocks: usize, n_byte_groups: usize, blocked_size: usize,
                codes_flat: &[Vec<u8>], _perm0: &[usize; 16]) -> Array1<u8> {
    // Sequential layout: each byte stored as-is, vectors in order.
    let mut blocked = vec![0u8; blocked_size];
    for block_idx in 0..n_blocks {
        let base_vec = block_idx * BLOCK;
        for g in 0..n_byte_groups {
            let out_offset = (block_idx * n_byte_groups + g) * BLOCK;
            for lane in 0..BLOCK {
                let vi = base_vec + lane;
                if vi < n {
                    blocked[out_offset + lane] = codes_flat[vi][g];
                }
            }
        }
    }
    Array1::from_vec(blocked)
}

// =============================================================================
// 4-bit NEON scoring with periodic uint16→float flushing
// =============================================================================

// EVOLVE-BLOCK-START
// Search pipeline: scoring kernel, LUT build, rotation, heap.
// Optimize for end-to-end search latency on 100K vectors, d=1536, 4-bit.
// The only entry point that must be preserved is search_inner() — its signature
// is called from the #[pyfunction] search wrapper below the EVOLVE-BLOCK.

#[cfg(target_arch = "aarch64")]
unsafe fn score_4bit_block_neon(
    blocked_codes: &[u8],
    uint8_luts: &[u8],
    block_offset: usize,
    n_byte_groups: usize,
    scale: f32,
    bias: f32,
    norms: &[f32],
    base_vec: usize,
    n_vectors: usize,
    out: &mut [f32; BLOCK],
) {
    use std::arch::aarch64::*;

    let mask = vdupq_n_u8(0x0F);
    let v_scale = vdupq_n_f32(scale);
    let n_batches = (n_byte_groups + FLUSH_EVERY - 1) / FLUSH_EVERY;

    // Float accumulators in NEON registers (8 × float32x4 = 32 floats)
    let mut fa = [vdupq_n_f32(0.0); 8];

    let codes_base = blocked_codes.as_ptr().add(block_offset);
    let luts_base = uint8_luts.as_ptr();

    for batch in 0..n_batches {
        let g_start = batch * FLUSH_EVERY;
        let g_end = (g_start + FLUSH_EVERY).min(n_byte_groups);
        let n_groups = g_end - g_start;

        let mut accum = [vdupq_n_u16(0); 4];

        // 4-group unrolled inner loop. Interleaves lookups to hide latency of vqtbl1q_u8
        let mut g = g_start;
        while g + 3 < g_end {
            let lp0 = luts_base.add(g * 32);
            let lp1 = luts_base.add((g + 1) * 32);
            let lp2 = luts_base.add((g + 2) * 32);
            let lp3 = luts_base.add((g + 3) * 32);
            let cp0 = codes_base.add(g * BLOCK);
            let cp1 = codes_base.add((g + 1) * BLOCK);
            let cp2 = codes_base.add((g + 2) * BLOCK);
            let cp3 = codes_base.add((g + 3) * BLOCK);

            for (lp, cp) in [(lp0, cp0), (lp1, cp1), (lp2, cp2), (lp3, cp3)] {
                let lut_hi = vld1q_u8(lp);
                let lut_lo = vld1q_u8(lp.add(16));
                let c0 = vld1q_u8(cp);
                let c1 = vld1q_u8(cp.add(16));
                let s0 = vaddq_u8(vqtbl1q_u8(lut_lo, vandq_u8(c0, mask)), vqtbl1q_u8(lut_hi, vshrq_n_u8(c0, 4)));
                let s1 = vaddq_u8(vqtbl1q_u8(lut_lo, vandq_u8(c1, mask)), vqtbl1q_u8(lut_hi, vshrq_n_u8(c1, 4)));
                accum[0] = vaddw_u8(accum[0], vget_low_u8(s0));
                accum[1] = vaddw_u8(accum[1], vget_high_u8(s0));
                accum[2] = vaddw_u8(accum[2], vget_low_u8(s1));
                accum[3] = vaddw_u8(accum[3], vget_high_u8(s1));
            }
            g += 4;
        }

        // Handle remaining groups (0-3)
        while g < g_end {
            let lp = luts_base.add(g * 32);
            let lut_hi = vld1q_u8(lp);
            let lut_lo = vld1q_u8(lp.add(16));
            let cp = codes_base.add(g * BLOCK);
            let c0 = vld1q_u8(cp);
            let c1 = vld1q_u8(cp.add(16));
            let s0 = vaddq_u8(vqtbl1q_u8(lut_lo, vandq_u8(c0, mask)),
                              vqtbl1q_u8(lut_hi, vshrq_n_u8(c0, 4)));
            let s1 = vaddq_u8(vqtbl1q_u8(lut_lo, vandq_u8(c1, mask)),
                              vqtbl1q_u8(lut_hi, vshrq_n_u8(c1, 4)));
            accum[0] = vaddw_u8(accum[0], vget_low_u8(s0));
            accum[1] = vaddw_u8(accum[1], vget_high_u8(s0));
            accum[2] = vaddw_u8(accum[2], vget_low_u8(s1));
            accum[3] = vaddw_u8(accum[3], vget_high_u8(s1));
            g += 1;
        }

        // Flush: uint16 → float via NEON widening + fused multiply-add
        let v_bias = vdupq_n_f32(n_groups as f32 * 2.0 * bias);
        for i in 0..4 {
            // Split uint16x8 into two uint32x4, convert to float32x4
            let lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(accum[i])));
            let hi = vcvtq_f32_u32(vmovl_u16(vget_high_u16(accum[i])));
            // fa += scale * val + bias
            fa[i * 2] = vaddq_f32(fa[i * 2], vfmaq_f32(v_bias, v_scale, lo));
            fa[i * 2 + 1] = vaddq_f32(fa[i * 2 + 1], vfmaq_f32(v_bias, v_scale, hi));
        }
    }

    // Write 32 scores to output buffer, applying norms
    let end = (base_vec + BLOCK).min(n_vectors);
    let out_ptr = out.as_mut_ptr();
    let norms_ptr = norms.as_ptr().add(base_vec);

    if end - base_vec == BLOCK {
        for i in 0..8 {
            let n = vld1q_f32(norms_ptr.add(i * 4));
            vst1q_f32(out_ptr.add(i * 4), vmulq_f32(fa[i], n));
        }
    } else {
        let mut float_accum = [0.0f32; BLOCK];
        for i in 0..8 {
            vst1q_f32(float_accum.as_mut_ptr().add(i * 4), fa[i]);
        }
        for lane in 0..BLOCK {
            *out_ptr.add(lane) = if lane < end - base_vec {
                float_accum[lane] * *norms_ptr.add(lane)
            } else {
                f32::NEG_INFINITY
            };
        }
    }
}

// =============================================================================
// AVX2 scoring kernel for x86_64
// =============================================================================

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn score_4bit_block_avx2(
    blocked_codes: &[u8],
    uint8_luts: &[u8],
    block_offset: usize,
    n_byte_groups: usize,
    scale: f32,
    bias: f32,
    norms: &[f32],
    base_vec: usize,
    n_vectors: usize,
    out: &mut [f32; BLOCK],
) {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    // FAISS-style reinterpret trick. 4 uint16 accumulators.
    // Inner loop: shuffle + reinterpret-as-uint16 add + srli. No add_epi8, no and-mask.
    // Carry correction at flush. uint16 wrapping is fine — correction recovers exact values.
    let mask = _mm256_set1_epi8(0x0F);
    let v_scale = _mm256_set1_ps(scale);
    let v_bias = _mm256_set1_ps(n_byte_groups as f32 * 2.0 * bias);

    // accu[0]/[1] = lo-nibble even/odd, accu[2]/[3] = hi-nibble even/odd
    let mut accu = [_mm256_setzero_si256(); 4];

    let codes_base = blocked_codes.as_ptr().add(block_offset);
    let luts_base = uint8_luts.as_ptr();

    let mut g = 0usize;
    while g + 3 < n_byte_groups {
        for gi in 0..4usize {
            let gg = g + gi;
            let lp = luts_base.add(gg * 32);
            let cp = codes_base.add(gg * BLOCK);

            let lut_hi = _mm256_broadcastsi128_si256(_mm_loadu_si128(lp as *const __m128i));
            let lut_lo = _mm256_broadcastsi128_si256(_mm_loadu_si128(lp.add(16) as *const __m128i));
            let codes = _mm256_loadu_si256(cp as *const __m256i);

            let clo = _mm256_and_si256(codes, mask);
            let chi = _mm256_and_si256(_mm256_srli_epi16(codes, 4), mask);

            let res0 = _mm256_shuffle_epi8(lut_lo, clo);
            let res1 = _mm256_shuffle_epi8(lut_hi, chi);

            accu[0] = _mm256_add_epi16(accu[0], res0);
            accu[1] = _mm256_add_epi16(accu[1], _mm256_srli_epi16(res0, 8));
            accu[2] = _mm256_add_epi16(accu[2], res1);
            accu[3] = _mm256_add_epi16(accu[3], _mm256_srli_epi16(res1, 8));
        }
        g += 4;
    }
    while g < n_byte_groups {
        let lp = luts_base.add(g * 32);
        let cp = codes_base.add(g * BLOCK);

        let lut_hi = _mm256_broadcastsi128_si256(_mm_loadu_si128(lp as *const __m128i));
        let lut_lo = _mm256_broadcastsi128_si256(_mm_loadu_si128(lp.add(16) as *const __m128i));
        let codes = _mm256_loadu_si256(cp as *const __m256i);

        let clo = _mm256_and_si256(codes, mask);
        let chi = _mm256_and_si256(_mm256_srli_epi16(codes, 4), mask);

        let res0 = _mm256_shuffle_epi8(lut_lo, clo);
        let res1 = _mm256_shuffle_epi8(lut_hi, chi);

        accu[0] = _mm256_add_epi16(accu[0], res0);
        accu[1] = _mm256_add_epi16(accu[1], _mm256_srli_epi16(res0, 8));
        accu[2] = _mm256_add_epi16(accu[2], res1);
        accu[3] = _mm256_add_epi16(accu[3], _mm256_srli_epi16(res1, 8));
        g += 1;
    }

    // Carry correction: even bytes accumulated carries into odd bytes.
    // accu[1] has correct odd scores (srli captured true hi bytes each time).
    // accu[0] has even + carry pollution. Fix: even = accu[0] - (accu[1] << 8)
    accu[0] = _mm256_sub_epi16(accu[0], _mm256_slli_epi16(accu[1], 8));
    accu[2] = _mm256_sub_epi16(accu[2], _mm256_slli_epi16(accu[3], 8));

    // Combine lo+hi nibble in float to avoid uint16 overflow
    let ae0_lo = _mm256_castsi256_si128(accu[0]);
    let ae0_hi = _mm256_extracti128_si256(accu[0], 1);
    let ao0_lo = _mm256_castsi256_si128(accu[1]);
    let ao0_hi = _mm256_extracti128_si256(accu[1], 1);
    let ae2_lo = _mm256_castsi256_si128(accu[2]);
    let ae2_hi = _mm256_extracti128_si256(accu[2], 1);
    let ao2_lo = _mm256_castsi256_si128(accu[3]);
    let ao2_hi = _mm256_extracti128_si256(accu[3], 1);

    // Interleave even/odd → sequential, for lo-nibble and hi-nibble separately
    let lo_01 = _mm_unpacklo_epi16(ae0_lo, ao0_lo);
    let lo_23 = _mm_unpackhi_epi16(ae0_lo, ao0_lo);
    let lo_45 = _mm_unpacklo_epi16(ae0_hi, ao0_hi);
    let lo_67 = _mm_unpackhi_epi16(ae0_hi, ao0_hi);
    let hi_01 = _mm_unpacklo_epi16(ae2_lo, ao2_lo);
    let hi_23 = _mm_unpackhi_epi16(ae2_lo, ao2_lo);
    let hi_45 = _mm_unpacklo_epi16(ae2_hi, ao2_hi);
    let hi_67 = _mm_unpackhi_epi16(ae2_hi, ao2_hi);

    // Convert to float and sum lo + hi contributions
    let f0 = _mm256_add_ps(
        _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(lo_01)),
        _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(hi_01)));
    let f1 = _mm256_add_ps(
        _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(lo_23)),
        _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(hi_23)));
    let f2 = _mm256_add_ps(
        _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(lo_45)),
        _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(hi_45)));
    let f3 = _mm256_add_ps(
        _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(lo_67)),
        _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(hi_67)));

    let end = (base_vec + BLOCK).min(n_vectors);
    let out_ptr = out.as_mut_ptr();
    let norms_ptr = norms.as_ptr().add(base_vec);
    if end - base_vec == BLOCK {
        for (i, f) in [f0, f1, f2, f3].iter().enumerate() {
            let scored = _mm256_fmadd_ps(v_scale, *f, v_bias);
            let n = _mm256_loadu_ps(norms_ptr.add(i * 8));
            _mm256_storeu_ps(out_ptr.add(i * 8), _mm256_mul_ps(scored, n));
        }
    } else {
        let mut float_accum = [0.0f32; BLOCK];
        for (i, f) in [f0, f1, f2, f3].iter().enumerate() {
            _mm256_storeu_ps(float_accum.as_mut_ptr().add(i * 8), _mm256_fmadd_ps(v_scale, *f, v_bias));
        }
        for lane in 0..BLOCK {
            *out_ptr.add(lane) = if lane < end - base_vec {
                float_accum[lane] * *norms_ptr.add(lane)
            } else {
                f32::NEG_INFINITY
            };
        }
    }
}

/// Fused multi-query scoring + heap top-k. Processes NQ=4 queries per block,
/// sharing code loads. No score array materialization — heap updated per block.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn search_multi_query_avx2(
    blocked_codes: &[u8],
    luts: &[&[u8]],
    scales: &[f32],
    biases: &[f32],
    n_byte_groups: usize,
    norms: &[f32],
    n_vectors: usize,
    nq: usize,
    k: usize,
    heap_scores: &mut [Vec<f32>],
    heap_indices: &mut [Vec<u32>],
    heap_sizes: &mut [usize],
    heap_mins: &mut [f32],
    heap_min_idxs: &mut [usize],
) {
    use std::arch::x86_64::*;

    let n_blocks = (n_vectors + BLOCK - 1) / BLOCK;
    let mask = _mm256_set1_epi8(0x0F);
    let codes_base = blocked_codes.as_ptr();

    for b in 0..n_blocks {
        let base_vec = b * BLOCK;
        let mut accus = [[_mm256_setzero_si256(); 4]; 4];

        for g in 0..n_byte_groups {
            let cp = codes_base.add((b * n_byte_groups + g) * BLOCK);
            let codes_v = _mm256_loadu_si256(cp as *const __m256i);
            let clo = _mm256_and_si256(codes_v, mask);
            let chi = _mm256_and_si256(_mm256_srli_epi16(codes_v, 4), mask);

            for qi in 0..4 {
                let lut = _mm256_loadu_si256(luts[qi].as_ptr().add(g * 32) as *const __m256i);
                let res0 = _mm256_shuffle_epi8(lut, clo);
                let res1 = _mm256_shuffle_epi8(lut, chi);
                accus[qi][0] = _mm256_add_epi16(accus[qi][0], res0);
                accus[qi][1] = _mm256_add_epi16(accus[qi][1], _mm256_srli_epi16(res0, 8));
                accus[qi][2] = _mm256_add_epi16(accus[qi][2], res1);
                accus[qi][3] = _mm256_add_epi16(accus[qi][3], _mm256_srli_epi16(res1, 8));
            }
        }

        let end = (base_vec + BLOCK).min(n_vectors);
        let norms_ptr = norms.as_ptr().add(base_vec);

        for qi in 0..nq {
            let v_scale = _mm256_set1_ps(scales[qi]);
            let v_bias = _mm256_set1_ps(n_byte_groups as f32 * 2.0 * biases[qi]);

            accus[qi][0] = _mm256_sub_epi16(accus[qi][0], _mm256_slli_epi16(accus[qi][1], 8));
            accus[qi][2] = _mm256_sub_epi16(accus[qi][2], _mm256_slli_epi16(accus[qi][3], 8));

            let dis0 = _mm256_add_epi16(
                _mm256_permute2x128_si256(accus[qi][0], accus[qi][1], 0x21),
                _mm256_blend_epi32(accus[qi][0], accus[qi][1], 0xF0),
            );
            let dis1 = _mm256_add_epi16(
                _mm256_permute2x128_si256(accus[qi][2], accus[qi][3], 0x21),
                _mm256_blend_epi32(accus[qi][2], accus[qi][3], 0xF0),
            );

            let mut block_out = [0.0f32; BLOCK];
            let bp = block_out.as_mut_ptr();
            let f0 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(dis0)));
            let f1 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_extracti128_si256(dis0, 1)));
            let f2 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(dis1)));
            let f3 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_extracti128_si256(dis1, 1)));

            if end - base_vec == BLOCK {
                for (i, f) in [f0, f1, f2, f3].iter().enumerate() {
                    let scored = _mm256_fmadd_ps(v_scale, *f, v_bias);
                    let n = _mm256_loadu_ps(norms_ptr.add(i * 8));
                    _mm256_storeu_ps(bp.add(i * 8), _mm256_mul_ps(scored, n));
                }
            } else {
                for (i, f) in [f0, f1, f2, f3].iter().enumerate() {
                    _mm256_storeu_ps(bp.add(i * 8), _mm256_fmadd_ps(v_scale, *f, v_bias));
                }
                for lane in 0..(end - base_vec) {
                    block_out[lane] *= *norms_ptr.add(lane);
                }
                for lane in (end - base_vec)..BLOCK {
                    block_out[lane] = f32::NEG_INFINITY;
                }
            }

            // Heap insertion with SIMD early rejection.
            // Check if any score in each 8-element chunk exceeds heap min.
            // Skip entire chunks where max(chunk) <= heap_min.
            let hs = &mut heap_scores[qi];
            let hi = &mut heap_indices[qi];
            let sz = &mut heap_sizes[qi];
            let hmin = &mut heap_mins[qi];
            let hmi = &mut heap_min_idxs[qi];

            if *sz < k {
                // Filling phase — just insert everything
                for lane in 0..(end - base_vec) {
                    hs[*sz] = block_out[lane];
                    hi[*sz] = (base_vec + lane) as u32;
                    *sz += 1;
                    if *sz == k {
                        *hmin = hs[0]; *hmi = 0;
                        for h in 1..k {
                            if hs[h] < *hmin { *hmin = hs[h]; *hmi = h; }
                        }
                    }
                }
            } else {
                // SIMD max check per 8-float chunk, skip if no candidates
                let v_hmin = _mm256_set1_ps(*hmin);
                for chunk in 0..4 {
                    let chunk_start = chunk * 8;
                    if chunk_start >= end - base_vec { break; }
                    let scores_v = _mm256_loadu_ps(block_out.as_ptr().add(chunk_start));
                    let cmp = _mm256_cmp_ps(scores_v, v_hmin, _CMP_GT_OQ);
                    if _mm256_movemask_ps(cmp) == 0 { continue; } // all <= min, skip

                    let chunk_end = (chunk_start + 8).min(end - base_vec);
                    for lane in chunk_start..chunk_end {
                        let score = block_out[lane];
                        if score > *hmin {
                            hs[*hmi] = score;
                            hi[*hmi] = (base_vec + lane) as u32;
                            // SIMD min-find over k=64 heap entries (8 chunks of 8)
                            let hp = hs.as_ptr();
                            let mut vmin = _mm256_loadu_ps(hp);
                            for c in 1..(k / 8) {
                                vmin = _mm256_min_ps(vmin, _mm256_loadu_ps(hp.add(c * 8)));
                            }
                            // Horizontal min of 8 floats
                            let lo = _mm256_castps256_ps128(vmin);
                            let hi128 = _mm256_extractf128_ps(vmin, 1);
                            let m4 = _mm_min_ps(lo, hi128);
                            let m2 = _mm_min_ps(m4, _mm_movehl_ps(m4, m4));
                            let m1 = _mm_min_ps(m2, _mm_shuffle_ps(m2, m2, 1));
                            *hmin = _mm_cvtss_f32(m1);
                            // Find index of min (scalar scan, but only after SIMD found the value)
                            *hmi = 0;
                            for h in 1..k {
                                if hs[h] < hs[*hmi] { *hmi = h; }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Repack 3-bit codes into two blocked arrays:
/// - sub_codes: 2-bit nibble format from planes 0,1
/// - plane2: packed bits blocked by 32 vectors
fn repack_3bit_inner(
    packed_codes: ArrayView2<u8>,
    dim: usize,
) -> (Array1<u8>, Array1<u8>, usize) {
    let n = packed_codes.nrows();
    let bytes_per_plane = dim / 8;
    let n_blocks = (n + BLOCK - 1) / BLOCK;

    let sub_byte_groups = dim / 4;
    let mut sub_codes = vec![0u8; n_blocks * sub_byte_groups * BLOCK];

    let plane2_byte_groups = bytes_per_plane;
    let mut plane2_blocked = vec![0u8; n_blocks * plane2_byte_groups * BLOCK];

    for block_idx in 0..n_blocks {
        let base_vec = block_idx * BLOCK;

        for g in 0..sub_byte_groups {
            let out_offset = (block_idx * sub_byte_groups + g) * BLOCK;
            for lane in 0..BLOCK {
                let vec_idx = base_vec + lane;
                if vec_idx >= n { continue; }

                let mut byte_val = 0u8;
                let dim_start = g * 4;
                for c in 0..4usize {
                    let j = dim_start + c;
                    let byte_in_plane = j / 8;
                    let bit_in_byte = 7 - (j % 8);
                    let mask = 1u8 << bit_in_byte;

                    let mut code = 0u8;
                    for p in 0..2usize {
                        let plane_byte = packed_codes[[vec_idx, p * bytes_per_plane + byte_in_plane]];
                        if plane_byte & mask != 0 { code |= 1 << p; }
                    }
                    byte_val |= code << ((3 - c) * 2);
                }
                sub_codes[out_offset + lane] = byte_val;
            }
        }

        for g in 0..plane2_byte_groups {
            let out_offset = (block_idx * plane2_byte_groups + g) * BLOCK;
            for lane in 0..BLOCK {
                let vec_idx = base_vec + lane;
                if vec_idx >= n { continue; }
                plane2_blocked[out_offset + lane] = packed_codes[[vec_idx, 2 * bytes_per_plane + g]];
            }
        }
    }

    (Array1::from_vec(sub_codes), Array1::from_vec(plane2_blocked), n_blocks)
}

/// Score one block for TWO queries, sharing code loads.
/// Loads each code byte once, looks up in both queries' LUTs.
#[cfg(target_arch = "aarch64")]
unsafe fn score_2query_block_neon(
    blocked_codes: &[u8],
    luts_a: &[u8],
    luts_b: &[u8],
    block_offset: usize,
    n_byte_groups: usize,
    scale_a: f32,
    bias_a: f32,
    scale_b: f32,
    bias_b: f32,
    norms: &[f32],
    base_vec: usize,
    n_vectors: usize,
    row_a: &mut [f32],
    row_b: &mut [f32],
) {
    use std::arch::aarch64::*;

    let mask = vdupq_n_u8(0x0F);
    let v_scale_a = vdupq_n_f32(scale_a);
    let v_scale_b = vdupq_n_f32(scale_b);
    let n_batches = (n_byte_groups + FLUSH_EVERY - 1) / FLUSH_EVERY;

    let mut fa_a = [vdupq_n_f32(0.0); 8];
    let mut fa_b = [vdupq_n_f32(0.0); 8];

    let codes_base = blocked_codes.as_ptr().add(block_offset);
    let luts_a_base = luts_a.as_ptr();
    let luts_b_base = luts_b.as_ptr();

    for batch in 0..n_batches {
        let g_start = batch * FLUSH_EVERY;
        let g_end = (g_start + FLUSH_EVERY).min(n_byte_groups);
        let n_groups = g_end - g_start;

        let mut acc_a = [vdupq_n_u16(0); 4];
        let mut acc_b = [vdupq_n_u16(0); 4];

        for g in g_start..g_end {
            // Load codes ONCE — shared between both queries
            let cp = codes_base.add(g * BLOCK);
            let c0 = vld1q_u8(cp);
            let c1 = vld1q_u8(cp.add(16));

            // Split nibbles ONCE — shared
            let lo0 = vandq_u8(c0, mask);
            let lo1 = vandq_u8(c1, mask);
            let hi0 = vshrq_n_u8(c0, 4);
            let hi1 = vshrq_n_u8(c1, 4);

            // Query A lookups
            let lp_a = luts_a_base.add(g * 32);
            let lut_hi_a = vld1q_u8(lp_a);
            let lut_lo_a = vld1q_u8(lp_a.add(16));
            let s0_a = vaddq_u8(vqtbl1q_u8(lut_lo_a, lo0), vqtbl1q_u8(lut_hi_a, hi0));
            let s1_a = vaddq_u8(vqtbl1q_u8(lut_lo_a, lo1), vqtbl1q_u8(lut_hi_a, hi1));
            acc_a[0] = vaddw_u8(acc_a[0], vget_low_u8(s0_a));
            acc_a[1] = vaddw_u8(acc_a[1], vget_high_u8(s0_a));
            acc_a[2] = vaddw_u8(acc_a[2], vget_low_u8(s1_a));
            acc_a[3] = vaddw_u8(acc_a[3], vget_high_u8(s1_a));

            // Query B lookups (reuses same codes, split nibbles)
            let lp_b = luts_b_base.add(g * 32);
            let lut_hi_b = vld1q_u8(lp_b);
            let lut_lo_b = vld1q_u8(lp_b.add(16));
            let s0_b = vaddq_u8(vqtbl1q_u8(lut_lo_b, lo0), vqtbl1q_u8(lut_hi_b, hi0));
            let s1_b = vaddq_u8(vqtbl1q_u8(lut_lo_b, lo1), vqtbl1q_u8(lut_hi_b, hi1));
            acc_b[0] = vaddw_u8(acc_b[0], vget_low_u8(s0_b));
            acc_b[1] = vaddw_u8(acc_b[1], vget_high_u8(s0_b));
            acc_b[2] = vaddw_u8(acc_b[2], vget_low_u8(s1_b));
            acc_b[3] = vaddw_u8(acc_b[3], vget_high_u8(s1_b));
        }

        // Flush query A
        let v_bias_a = vdupq_n_f32(n_groups as f32 * 2.0 * bias_a);
        for i in 0..4 {
            let lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(acc_a[i])));
            let hi = vcvtq_f32_u32(vmovl_u16(vget_high_u16(acc_a[i])));
            fa_a[i * 2] = vaddq_f32(fa_a[i * 2], vfmaq_f32(v_bias_a, v_scale_a, lo));
            fa_a[i * 2 + 1] = vaddq_f32(fa_a[i * 2 + 1], vfmaq_f32(v_bias_a, v_scale_a, hi));
        }

        // Flush query B
        let v_bias_b = vdupq_n_f32(n_groups as f32 * 2.0 * bias_b);
        for i in 0..4 {
            let lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(acc_b[i])));
            let hi = vcvtq_f32_u32(vmovl_u16(vget_high_u16(acc_b[i])));
            fa_b[i * 2] = vaddq_f32(fa_b[i * 2], vfmaq_f32(v_bias_b, v_scale_b, lo));
            fa_b[i * 2 + 1] = vaddq_f32(fa_b[i * 2 + 1], vfmaq_f32(v_bias_b, v_scale_b, hi));
        }
    }

    // Write with norms
    let end = (base_vec + BLOCK).min(n_vectors);
    let row_a_ptr = row_a.as_mut_ptr().add(base_vec);
    let row_b_ptr = row_b.as_mut_ptr().add(base_vec);
    let norms_ptr = norms.as_ptr().add(base_vec);

    if end - base_vec == BLOCK {
        for i in 0..8 {
            let n = vld1q_f32(norms_ptr.add(i * 4));
            vst1q_f32(row_a_ptr.add(i * 4), vmulq_f32(fa_a[i], n));
            vst1q_f32(row_b_ptr.add(i * 4), vmulq_f32(fa_b[i], n));
        }
    } else {
        let mut buf_a = [0.0f32; BLOCK];
        let mut buf_b = [0.0f32; BLOCK];
        for i in 0..8 {
            vst1q_f32(buf_a.as_mut_ptr().add(i * 4), fa_a[i]);
            vst1q_f32(buf_b.as_mut_ptr().add(i * 4), fa_b[i]);
        }
        for lane in 0..(end - base_vec) {
            *row_a_ptr.add(lane) = buf_a[lane] * *norms_ptr.add(lane);
            *row_b_ptr.add(lane) = buf_b[lane] * *norms_ptr.add(lane);
        }
    }
}

/// Score one block for FOUR queries, sharing code loads and nibble splits.
/// Codes loaded once, nibbles split once, then looked up in 4 different LUTs.
#[cfg(target_arch = "aarch64")]
unsafe fn score_4query_block_neon(
    blocked_codes: &[u8],
    luts: [&[u8]; 4],
    block_offset: usize,
    n_byte_groups: usize,
    scales: [f32; 4],
    biases: [f32; 4],
    norms: &[f32],
    base_vec: usize,
    n_vectors: usize,
    rows: [*mut f32; 4],
) {
    use std::arch::aarch64::*;

    let mask = vdupq_n_u8(0x0F);
    let n_batches = (n_byte_groups + FLUSH_EVERY - 1) / FLUSH_EVERY;

    // Float accumulators on stack (flushed infrequently)
    let mut fa: [[float32x4_t; 8]; 4] = [[vdupq_n_f32(0.0); 8]; 4];

    let codes_base = blocked_codes.as_ptr().add(block_offset);

    for batch in 0..n_batches {
        let g_start = batch * FLUSH_EVERY;
        let g_end = (g_start + FLUSH_EVERY).min(n_byte_groups);
        let n_groups = g_end - g_start;

        let mut acc: [[uint16x8_t; 4]; 4] = [[vdupq_n_u16(0); 4]; 4];

        for g in g_start..g_end {
            // Load codes ONCE
            let cp = codes_base.add(g * BLOCK);
            let c0 = vld1q_u8(cp);
            let c1 = vld1q_u8(cp.add(16));

            // Split nibbles ONCE
            let lo0 = vandq_u8(c0, mask);
            let lo1 = vandq_u8(c1, mask);
            let hi0 = vshrq_n_u8(c0, 4);
            let hi1 = vshrq_n_u8(c1, 4);

            // Score 4 queries against the same nibbles
            for q in 0..4 {
                let lp = luts[q].as_ptr().add(g * 32);
                let lut_hi = vld1q_u8(lp);
                let lut_lo = vld1q_u8(lp.add(16));
                let s0 = vaddq_u8(vqtbl1q_u8(lut_lo, lo0), vqtbl1q_u8(lut_hi, hi0));
                let s1 = vaddq_u8(vqtbl1q_u8(lut_lo, lo1), vqtbl1q_u8(lut_hi, hi1));
                acc[q][0] = vaddw_u8(acc[q][0], vget_low_u8(s0));
                acc[q][1] = vaddw_u8(acc[q][1], vget_high_u8(s0));
                acc[q][2] = vaddw_u8(acc[q][2], vget_low_u8(s1));
                acc[q][3] = vaddw_u8(acc[q][3], vget_high_u8(s1));
            }
        }

        // Flush each query
        for q in 0..4 {
            let v_scale = vdupq_n_f32(scales[q]);
            let v_bias = vdupq_n_f32(n_groups as f32 * 2.0 * biases[q]);
            for i in 0..4 {
                let lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(acc[q][i])));
                let hi = vcvtq_f32_u32(vmovl_u16(vget_high_u16(acc[q][i])));
                fa[q][i * 2] = vaddq_f32(fa[q][i * 2], vfmaq_f32(v_bias, v_scale, lo));
                fa[q][i * 2 + 1] = vaddq_f32(fa[q][i * 2 + 1], vfmaq_f32(v_bias, v_scale, hi));
            }
        }
    }

    // Write with norms
    let end = (base_vec + BLOCK).min(n_vectors);
    let norms_ptr = norms.as_ptr().add(base_vec);

    for q in 0..4 {
        let rp = rows[q].add(base_vec);
        if end - base_vec == BLOCK {
            for i in 0..8 {
                let n = vld1q_f32(norms_ptr.add(i * 4));
                vst1q_f32(rp.add(i * 4), vmulq_f32(fa[q][i], n));
            }
        } else {
            let mut buf = [0.0f32; BLOCK];
            for i in 0..8 {
                vst1q_f32(buf.as_mut_ptr().add(i * 4), fa[q][i]);
            }
            for lane in 0..(end - base_vec) {
                *rp.add(lane) = buf[lane] * *norms_ptr.add(lane);
            }
        }
    }
}

/// Per-query nibble LUTs for NEON scoring (works for 2-bit and 4-bit).
struct QueryNeonLut {
    uint8_luts: Vec<u8>,  // n_byte_groups * 32 bytes: [hi_16 | lo_16] per group
    scale: f32,
    bias: f32,
}

/// Build nibble LUTs for NEON scoring.
///
/// For any bit width where codes_per_byte is even, split each byte into
/// two nibbles. Each nibble encodes codes_per_nibble codes of `bits` bits
/// each, giving 2^4 = 16 possible values — perfect for vqtbl1q_u8.
///
/// 2-bit: 4 codes/byte → 2 codes/nibble → nibble = code0<<2 | code1
/// 4-bit: 2 codes/byte → 1 code/nibble → nibble = code0
fn build_query_neon_lut(
    q_rot: ArrayView2<f32>,
    qi: usize,
    centroids: ArrayView1<f32>,
    bits: usize,
    dim: usize,
) -> QueryNeonLut {
    let row = q_rot.row(qi);
    build_query_neon_lut_from_slice(row.as_slice().unwrap(), centroids, bits, dim)
}

fn build_query_neon_lut_from_slice(
    q_rot_row: &[f32],
    centroids: ArrayView1<f32>,
    bits: usize,
    dim: usize,
) -> QueryNeonLut {
    let codes_per_byte = 8 / bits;
    let codes_per_nibble = codes_per_byte / 2;
    let n_byte_groups = dim / codes_per_byte;
    let code_mask = (1u16 << bits) - 1;

    let mut uint8_luts = vec![0u8; n_byte_groups * 32];
    let mut float_vals = vec![0.0f32; n_byte_groups * 32]; // 16 hi + 16 lo per group
    let mut global_min = f32::MAX;
    let mut global_max = f32::MIN;

    for g in 0..n_byte_groups {
        let dim_start = g * codes_per_byte;

        // High nibble: first codes_per_nibble codes
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

        // Low nibble: last codes_per_nibble codes
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
    // On x86, max_lut ensures combine2x2 (adding two sub-quantizer totals) fits in uint16.
    // On ARM, no combine2x2 — use full 7-bit range (127) for best precision.
    #[cfg(target_arch = "x86_64")]
    let max_lut = (65535.0 / (n_byte_groups as f64 * 2.0)).floor().min(127.0) as f32;
    #[cfg(not(target_arch = "x86_64"))]
    let max_lut = 127.0f32;
    let scale = if range > 1e-10 { range / max_lut } else { 1.0 };
    let bias = global_min;

    for i in 0..float_vals.len() {
        uint8_luts[i] = ((float_vals[i] - bias) / scale).round().min(max_lut) as u8;
    }

    QueryNeonLut { uint8_luts, scale, bias }
}

/// NEON-accelerated scoring for 2-bit and 4-bit codes.
fn score_neon_inner(
    q_rot: ArrayView2<f32>,
    blocked_codes: &[u8],
    centroids: ArrayView1<f32>,
    norms: ArrayView1<f32>,
    bits: usize,
    dim: usize,
    n_vectors: usize,
    n_blocks: usize,
) -> Array2<f32> {
    let nq = q_rot.nrows();
    let codes_per_byte = 8 / bits;
    let n_byte_groups = dim / codes_per_byte;
    let norms_slice = norms.as_slice().unwrap();

    // Prebuild all LUTs in parallel
    let query_luts: Vec<QueryNeonLut> = (0..nq)
        .into_par_iter()
        .map(|qi| build_query_neon_lut(q_rot, qi, centroids, bits, dim))
        .collect();

    // Block-parallel scoring: each thread owns a disjoint range of blocks.
    // For each block, score ALL queries. Each block's code data is loaded
    // once and reused across all queries — shared memory reads.
    // Threads write to non-overlapping vector ranges, so no synchronization needed.
    let mut flat_scores = vec![0.0f32; nq * n_vectors];

    // Shared pointer for parallel write to disjoint regions.
    // Safe because each block_idx writes to base_vec..base_vec+BLOCK,
    // which is non-overlapping across different block_idx values.
    let scores_base = std::sync::atomic::AtomicPtr::new(flat_scores.as_mut_ptr());

    (0..n_blocks).into_par_iter().for_each(|block_idx| {
        let base_vec = block_idx * BLOCK;
        let block_offset = block_idx * n_byte_groups * BLOCK;
        let ptr = scores_base.load(std::sync::atomic::Ordering::Relaxed);

        // Process queries in groups of 4 with shared code loads
        let mut qi = 0;
        while qi + 3 < nq {
            unsafe {

                #[cfg(target_arch = "aarch64")]
                score_4query_block_neon(
                    blocked_codes,
                    [&query_luts[qi].uint8_luts, &query_luts[qi+1].uint8_luts,
                     &query_luts[qi+2].uint8_luts, &query_luts[qi+3].uint8_luts],
                    block_offset, n_byte_groups,
                    [query_luts[qi].scale, query_luts[qi+1].scale,
                     query_luts[qi+2].scale, query_luts[qi+3].scale],
                    [query_luts[qi].bias, query_luts[qi+1].bias,
                     query_luts[qi+2].bias, query_luts[qi+3].bias],
                    norms_slice, base_vec, n_vectors,
                    [ptr.add(qi * n_vectors), ptr.add((qi+1) * n_vectors),
                     ptr.add((qi+2) * n_vectors), ptr.add((qi+3) * n_vectors)],
                );

                #[cfg(not(target_arch = "aarch64"))]
                {
                    // x86_64: use AVX2 for each query individually
                    for q in qi..qi+4 {
                        let qlut = &query_luts[q];
                        let mut block_out = [0.0f32; BLOCK];
                        #[cfg(target_arch = "x86_64")]
                        {
                            if is_x86_feature_detected!("avx2") {
                                score_4bit_block_avx2(
                                    blocked_codes, &qlut.uint8_luts, block_offset, n_byte_groups,
                                    qlut.scale, qlut.bias, norms_slice, base_vec, n_vectors, &mut block_out,
                                );
                            }
                        }
                        let row = std::slice::from_raw_parts_mut(ptr.add(q * n_vectors), n_vectors);
                        let end = (base_vec + BLOCK).min(n_vectors);
                        for lane in 0..(end - base_vec) {
                            *row.get_unchecked_mut(base_vec + lane) = block_out[lane];
                        }
                    }
                }
            }
            qi += 4;
        }

        // Handle remaining 1-3 queries
        while qi + 1 < nq {
            unsafe {
                #[cfg(target_arch = "aarch64")]
                score_2query_block_neon(
                    blocked_codes, &query_luts[qi].uint8_luts, &query_luts[qi+1].uint8_luts,
                    block_offset, n_byte_groups,
                    query_luts[qi].scale, query_luts[qi].bias,
                    query_luts[qi+1].scale, query_luts[qi+1].bias,
                    norms_slice, base_vec, n_vectors,
                    std::slice::from_raw_parts_mut(ptr.add(qi * n_vectors), n_vectors),
                    std::slice::from_raw_parts_mut(ptr.add((qi+1) * n_vectors), n_vectors),
                );

                #[cfg(not(target_arch = "aarch64"))]
                {
                    for q in qi..qi+2 {
                        let qlut = &query_luts[q];
                        let mut block_out = [0.0f32; BLOCK];
                        #[cfg(target_arch = "x86_64")]
                        {
                            if is_x86_feature_detected!("avx2") {
                                score_4bit_block_avx2(
                                    blocked_codes, &qlut.uint8_luts, block_offset, n_byte_groups,
                                    qlut.scale, qlut.bias, norms_slice, base_vec, n_vectors, &mut block_out,
                                );
                            }
                        }
                        let row = std::slice::from_raw_parts_mut(ptr.add(q * n_vectors), n_vectors);
                        let end = (base_vec + BLOCK).min(n_vectors);
                        for lane in 0..(end - base_vec) {
                            *row.get_unchecked_mut(base_vec + lane) = block_out[lane];
                        }
                    }
                }
            }
            qi += 2;
        }

        if qi < nq {
            let qlut = &query_luts[qi];
            unsafe {
                let row = std::slice::from_raw_parts_mut(
                    ptr.add(qi * n_vectors),
                    n_vectors,
                );

                #[cfg(target_arch = "aarch64")]
                {
                    let mut block_out = [0.0f32; BLOCK];
                    score_4bit_block_neon(
                        blocked_codes,
                        &qlut.uint8_luts,
                        block_offset,
                        n_byte_groups,
                        qlut.scale,
                        qlut.bias,
                        norms_slice,
                        base_vec,
                        n_vectors,
                        &mut block_out,
                    );
                    let end = (base_vec + BLOCK).min(n_vectors);
                    for lane in 0..(end - base_vec) {
                        *row.get_unchecked_mut(base_vec + lane) = block_out[lane];
                    }
                }

                #[cfg(not(target_arch = "aarch64"))]
                {
                    let mut block_out = [0.0f32; BLOCK];
                    #[cfg(target_arch = "x86_64")]
                    {
                        if is_x86_feature_detected!("avx2") {
                            score_4bit_block_avx2(
                                blocked_codes, &qlut.uint8_luts, block_offset, n_byte_groups,
                                qlut.scale, qlut.bias, norms_slice, base_vec, n_vectors, &mut block_out,
                            );
                        }
                    }
                    let end = (base_vec + BLOCK).min(n_vectors);
                    for lane in 0..(end - base_vec) {
                        *row.get_unchecked_mut(base_vec + lane) = block_out[lane];
                    }
                }
            }
        }
    });

    let scores = Array2::from_shape_vec((nq, n_vectors), flat_scores).unwrap();

    scores
}

/// Generic scoring using byte-level LUTs (works for any bit width).
fn score_generic_inner(
    q_rot: ArrayView2<f32>,
    blocked_codes: &[u8],
    centroids: ArrayView1<f32>,
    norms: ArrayView1<f32>,
    bits: usize,
    dim: usize,
    n_vectors: usize,
    n_blocks: usize,
) -> Array2<f32> {
    let nq = q_rot.nrows();
    let codes_per_byte = 8 / bits;
    let n_byte_groups = dim / codes_per_byte;
    let shift_bits = bits;

    let rows: Vec<Vec<f32>> = (0..nq)
        .into_par_iter()
        .map(|qi| {
            let mut byte_lut = vec![0.0f32; n_byte_groups * 256];

            for g in 0..n_byte_groups {
                let dim_start = g * codes_per_byte;
                for v in 0u16..256 {
                    let mut s = 0.0f32;
                    for c in 0..codes_per_byte {
                        let shift = (codes_per_byte - 1 - c) * shift_bits;
                        let code = ((v as u8) >> shift) & ((1 << bits) - 1);
                        let j = dim_start + c;
                        s += q_rot[[qi, j]] * centroids[code as usize];
                    }
                    byte_lut[g * 256 + v as usize] = s;
                }
            }

            let mut row = vec![0.0f32; n_vectors];

            for block_idx in 0..n_blocks {
                let base_vec = block_idx * BLOCK;
                let block_offset = block_idx * n_byte_groups * BLOCK;
                let mut block_scores = [0.0f32; BLOCK];

                for g in 0..n_byte_groups {
                    let codes_start = block_offset + g * BLOCK;
                    let lut_base = &byte_lut[g * 256..g * 256 + 256];

                    for lane in 0..BLOCK {
                        unsafe {
                            let byte_val =
                                *blocked_codes.get_unchecked(codes_start + lane) as usize;
                            *block_scores.get_unchecked_mut(lane) +=
                                *lut_base.get_unchecked(byte_val);
                        }
                    }
                }

                let end = (base_vec + BLOCK).min(n_vectors);
                for lane in 0..(end - base_vec) {
                    row[base_vec + lane] = block_scores[lane] * norms[base_vec + lane];
                }
            }

            row
        })
        .collect();

    let mut scores = Array2::<f32>::zeros((nq, n_vectors));
    for (qi, row) in rows.into_iter().enumerate() {
        scores.row_mut(qi).as_slice_mut().unwrap().copy_from_slice(&row);
    }

    scores
}

#[pymodule]
mod py_turboquant {
    use super::*;

    #[pyfunction]
    fn repack<'py>(
        py: Python<'py>,
        packed_codes: PyReadonlyArray2<u8>,
        bits: usize,
        dim: usize,
    ) -> (Bound<'py, PyArray1<u8>>, usize) {
        let packed_codes = packed_codes.as_array();
        let (blocked, n_blocks) = repack_inner(packed_codes, bits, dim);
        (blocked.into_pyarray(py), n_blocks)
    }

    #[pyfunction]
    fn score<'py>(
        py: Python<'py>,
        q_rot: PyReadonlyArray2<f32>,
        blocked_codes: PyReadonlyArray1<u8>,
        centroids: PyReadonlyArray1<f32>,
        norms: PyReadonlyArray1<f32>,
        bits: usize,
        dim: usize,
        n_vectors: usize,
        n_blocks: usize,
    ) -> Bound<'py, PyArray2<f32>> {
        let q_rot = q_rot.as_array();
        let blocked_codes = blocked_codes.as_array();
        let centroids = centroids.as_array();
        let norms = norms.as_array();
        let codes_slice = blocked_codes.as_slice().unwrap();

        let result = if bits == 2 || bits == 3 || bits == 4 {
            score_neon_inner(
                q_rot, codes_slice, centroids, norms, bits, dim, n_vectors, n_blocks,
            )
        } else {
            score_generic_inner(
                q_rot, codes_slice, centroids, norms, bits, dim, n_vectors, n_blocks,
            )
        };

        result.into_pyarray(py)
    }

    /// Fused scoring + heap top-k. No full scores matrix allocated.
    #[pyfunction]
    fn score_topk<'py>(
        py: Python<'py>,
        q_rot: PyReadonlyArray2<f32>,
        blocked_codes: PyReadonlyArray1<u8>,
        centroids: PyReadonlyArray1<f32>,
        norms: PyReadonlyArray1<f32>,
        bits: usize,
        dim: usize,
        n_vectors: usize,
        n_blocks: usize,
        k: usize,
    ) -> (Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<i64>>) {
        let q_rot = q_rot.as_array();
        let blocked_codes = blocked_codes.as_array();
        let centroids = centroids.as_array();
        let norms = norms.as_array();
        let codes_slice = blocked_codes.as_slice().unwrap();
        let norms_slice = norms.as_slice().unwrap();
        let nq = q_rot.nrows();
        let k = k.min(n_vectors);
        let codes_per_byte = 8 / bits;
        let n_byte_groups = dim / codes_per_byte;

        // QBS (Query Batch Scoring): parallel over query batches.
        // Each thread takes a batch of queries, scans ALL blocks once,
        // and for each block scores all queries in the batch (sharing
        // the code load). Each query maintains its own k-element heap.
        const QBS: usize = 4; // queries per batch

        // Prebuild all query LUTs
        let query_luts: Vec<QueryNeonLut> = (0..nq)
            .into_par_iter()
            .map(|qi| build_query_neon_lut(q_rot, qi, centroids, bits, dim))
            .collect();

        let results: Vec<Vec<(Vec<f32>, Vec<i64>)>> = (0..nq)
            .step_by(QBS)
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|qi_start| {
                let qi_end = (qi_start + QBS).min(nq);
                let batch_size = qi_end - qi_start;

                // Per-query heaps
                let mut heap_scores: Vec<Vec<f32>> = (0..batch_size)
                    .map(|_| vec![f32::NEG_INFINITY; k])
                    .collect();
                let mut heap_indices: Vec<Vec<u32>> = (0..batch_size)
                    .map(|_| vec![0u32; k])
                    .collect();
                let mut heap_sizes = vec![0usize; batch_size];
                let mut heap_mins = vec![f32::NEG_INFINITY; batch_size];
                let mut heap_min_idxs = vec![0usize; batch_size];

                for block_idx in 0..n_blocks {
                    let base_vec = block_idx * BLOCK;
                    let block_offset = block_idx * n_byte_groups * BLOCK;

                    // Score this block for all queries in the batch.
                    // Code block loaded once into L1, shared across queries.
                    for qi_off in 0..batch_size {
                        let qi = qi_start + qi_off;
                        let qlut = &query_luts[qi];

                        // Stack-allocated 32-element output — zero heap allocation
                        let mut block_out = [0.0f32; BLOCK];

                        unsafe {
                            #[cfg(target_arch = "aarch64")]
                            score_4bit_block_neon(
                                codes_slice, &qlut.uint8_luts, block_offset, n_byte_groups,
                                qlut.scale, qlut.bias, norms_slice, base_vec, n_vectors, &mut block_out,
                            );

                            #[cfg(target_arch = "x86_64")]
                            {
                                if is_x86_feature_detected!("avx2") {
                                    score_4bit_block_avx2(
                                        codes_slice, &qlut.uint8_luts, block_offset, n_byte_groups,
                                        qlut.scale, qlut.bias, norms_slice, base_vec, n_vectors, &mut block_out,
                                    );
                                }
                            }
                        }

                        // Insert into this query's heap
                        let end = (base_vec + BLOCK).min(n_vectors);
                        for lane in 0..(end - base_vec) {
                            let score = block_out[lane];
                            if heap_sizes[qi_off] < k {
                                heap_scores[qi_off][heap_sizes[qi_off]] = score;
                                heap_indices[qi_off][heap_sizes[qi_off]] = (base_vec + lane) as u32;
                                heap_sizes[qi_off] += 1;
                                if heap_sizes[qi_off] == k {
                                    heap_mins[qi_off] = heap_scores[qi_off][0];
                                    heap_min_idxs[qi_off] = 0;
                                    for h in 1..k {
                                        if heap_scores[qi_off][h] < heap_mins[qi_off] {
                                            heap_mins[qi_off] = heap_scores[qi_off][h];
                                            heap_min_idxs[qi_off] = h;
                                        }
                                    }
                                }
                            } else if score > heap_mins[qi_off] {
                                let mi = heap_min_idxs[qi_off];
                                heap_scores[qi_off][mi] = score;
                                heap_indices[qi_off][mi] = (base_vec + lane) as u32;
                                heap_mins[qi_off] = heap_scores[qi_off][0];
                                heap_min_idxs[qi_off] = 0;
                                for h in 1..k {
                                    if heap_scores[qi_off][h] < heap_mins[qi_off] {
                                        heap_mins[qi_off] = heap_scores[qi_off][h];
                                        heap_min_idxs[qi_off] = h;
                                    }
                                }
                            }
                        }
                    }
                }

                // Sort each query's heap
                (0..batch_size)
                    .map(|qi_off| {
                        let sz = heap_sizes[qi_off];
                        let mut pairs: Vec<(f32, u32)> = heap_scores[qi_off][..sz]
                            .iter()
                            .zip(heap_indices[qi_off][..sz].iter())
                            .map(|(&s, &i)| (s, i))
                            .collect();
                        pairs.sort_unstable_by(|a, b| {
                            b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
                        });
                        let s: Vec<f32> = pairs.iter().map(|p| p.0).collect();
                        let i: Vec<i64> = pairs.iter().map(|p| p.1 as i64).collect();
                        (s, i)
                    })
                    .collect()
            })
            .collect();

        // Flatten batch results
        let results: Vec<(Vec<f32>, Vec<i64>)> = results.into_iter().flatten().collect();

        let mut top_scores = Array2::<f32>::zeros((nq, k));
        let mut top_indices = Array2::<i64>::zeros((nq, k));
        for (qi, (s, i)) in results.into_iter().enumerate() {
            for j in 0..s.len().min(k) {
                top_scores[[qi, j]] = s[j];
                top_indices[[qi, j]] = i[j];
            }
        }

        (top_scores.into_pyarray(py), top_indices.into_pyarray(py))
    }

    #[pyfunction]
    fn search<'py>(
        py: Python<'py>,
        queries: PyReadonlyArray2<f32>,
        rotation: PyReadonlyArray2<f32>,
        blocked_codes: PyReadonlyArray1<u8>,
        centroids: PyReadonlyArray1<f32>,
        norms: PyReadonlyArray1<f32>,
        bits: usize,
        dim: usize,
        n_vectors: usize,
        n_blocks: usize,
        k: usize,
    ) -> (Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<i64>>) {
        let (top_scores, top_indices) = search_inner(
            queries.as_array(), rotation.as_array(),
            blocked_codes.as_array(), centroids.as_array(), norms.as_array(),
            bits, dim, n_vectors, n_blocks, k,
        );
        (top_scores.into_pyarray(py), top_indices.into_pyarray(py))
    }
}

/// Full search: rotation + LUT build + scoring + heap top-k.
/// This is the main function to optimize for end-to-end search latency.
fn search_inner(
    queries: ArrayView2<f32>,
    rotation: ArrayView2<f32>,
    blocked_codes: ArrayView1<u8>,
    centroids: ArrayView1<f32>,
    norms: ArrayView1<f32>,
    bits: usize,
    dim: usize,
    n_vectors: usize,
    n_blocks: usize,
    k: usize,
) -> (Array2<f32>, Array2<i64>) {
    let codes_slice = blocked_codes.as_slice().unwrap();
    let norms_slice = norms.as_slice().unwrap();
    let nq = queries.nrows();
    let k = k.min(n_vectors);
    let n_byte_groups = dim / (8 / bits);

    let t_start = std::time::Instant::now();

    // Parallel rotation
    use numpy::ndarray::s;
    let rot_t = rotation.t();
    let n_threads = rayon::current_num_threads().max(1);
    let chunk = (nq + n_threads - 1) / n_threads;
    let q_rot_chunks: Vec<Array2<f32>> = (0..nq)
        .step_by(chunk)
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|start| {
            let end = (start + chunk).min(nq);
            queries.slice(s![start..end, ..]).dot(&rot_t)
        })
        .collect();

    let mut q_rot_flat = Vec::with_capacity(nq * dim);
    for c in &q_rot_chunks {
        q_rot_flat.extend_from_slice(c.as_slice().unwrap());
    }

    let t_rot = t_start.elapsed().as_micros();

    let query_luts: Vec<QueryNeonLut> = (0..nq)
        .into_par_iter()
        .map(|qi| {
            let row = &q_rot_flat[qi * dim..(qi + 1) * dim];
            build_query_neon_lut_from_slice(row, centroids, bits, dim)
        })
        .collect();

    let t_lut = t_start.elapsed().as_micros() - t_rot;
    let t_score_start = std::time::Instant::now();

    // Platform-specific scoring + top-k.
    #[cfg(target_arch = "aarch64")]
    let results = {
        // ARM: per-block scoring with inline heap (LUT stays in L1 naturally via vqtbl1q_u8)
        const QBS: usize = 4;
        let results: Vec<Vec<(Vec<f32>, Vec<i64>)>> = (0..nq)
            .step_by(QBS)
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|qi_start| {
                let qi_end = (qi_start + QBS).min(nq);
                let batch_size = qi_end - qi_start;

                let mut heap_scores: Vec<Vec<f32>> = (0..batch_size)
                    .map(|_| vec![f32::NEG_INFINITY; k])
                    .collect();
                let mut heap_indices: Vec<Vec<u32>> = (0..batch_size)
                    .map(|_| vec![0u32; k])
                    .collect();
                let mut heap_sizes = vec![0usize; batch_size];
                let mut heap_mins = vec![f32::NEG_INFINITY; batch_size];
                let mut heap_min_idxs = vec![0usize; batch_size];

                for block_idx in 0..n_blocks {
                    let base_vec = block_idx * BLOCK;
                    let block_offset = block_idx * n_byte_groups * BLOCK;

                    for qi_off in 0..batch_size {
                        let qi = qi_start + qi_off;
                        let qlut = &query_luts[qi];
                        let mut block_out = [0.0f32; BLOCK];

                        unsafe {
                            score_4bit_block_neon(
                                codes_slice, &qlut.uint8_luts, block_offset, n_byte_groups,
                                qlut.scale, qlut.bias, norms_slice, base_vec, n_vectors, &mut block_out,
                            );
                        }

                        let end = (base_vec + BLOCK).min(n_vectors);
                        for lane in 0..(end - base_vec) {
                            let score = block_out[lane];
                            if heap_sizes[qi_off] < k {
                                heap_scores[qi_off][heap_sizes[qi_off]] = score;
                                heap_indices[qi_off][heap_sizes[qi_off]] = (base_vec + lane) as u32;
                                heap_sizes[qi_off] += 1;
                                if heap_sizes[qi_off] == k {
                                    heap_mins[qi_off] = heap_scores[qi_off][0];
                                    heap_min_idxs[qi_off] = 0;
                                    for h in 1..k {
                                        if heap_scores[qi_off][h] < heap_mins[qi_off] {
                                            heap_mins[qi_off] = heap_scores[qi_off][h];
                                            heap_min_idxs[qi_off] = h;
                                        }
                                    }
                                }
                            } else if score > heap_mins[qi_off] {
                                let mi = heap_min_idxs[qi_off];
                                heap_scores[qi_off][mi] = score;
                                heap_indices[qi_off][mi] = (base_vec + lane) as u32;
                                heap_mins[qi_off] = heap_scores[qi_off][0];
                                heap_min_idxs[qi_off] = 0;
                                for h in 1..k {
                                    if heap_scores[qi_off][h] < heap_mins[qi_off] {
                                        heap_mins[qi_off] = heap_scores[qi_off][h];
                                        heap_min_idxs[qi_off] = h;
                                    }
                                }
                            }
                        }
                    }
                }

                (0..batch_size)
                    .map(|qi_off| {
                        let sz = heap_sizes[qi_off];
                        let mut pairs: Vec<(f32, u32)> = heap_scores[qi_off][..sz]
                            .iter()
                            .zip(heap_indices[qi_off][..sz].iter())
                            .map(|(&s, &i)| (s, i))
                            .collect();
                        pairs.sort_unstable_by(|a, b| {
                            b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
                        });
                        let s: Vec<f32> = pairs.iter().map(|p| p.0).collect();
                        let i: Vec<i64> = pairs.iter().map(|p| p.1 as i64).collect();
                        (s, i)
                    })
                    .collect()
            })
            .collect();
        results.into_iter().flatten().collect::<Vec<_>>()
    };

    #[cfg(target_arch = "x86_64")]
    let results = {
        // Fused multi-query scoring + heap: batch NQ=4 queries, share code loads.
        const NQ_BATCH: usize = 4;

        let results: Vec<(Vec<f32>, Vec<i64>)> = (0..nq)
            .step_by(NQ_BATCH)
            .collect::<Vec<_>>()
            .into_par_iter()
            .flat_map(|qi_start| {
                let qi_end = (qi_start + NQ_BATCH).min(nq);
                let batch_nq = qi_end - qi_start;

                let pad_qi = qi_end - 1;
                let lut_refs: Vec<&[u8]> = (0..NQ_BATCH)
                    .map(|i| {
                        let qi = if qi_start + i < qi_end { qi_start + i } else { pad_qi };
                        query_luts[qi].uint8_luts.as_slice()
                    })
                    .collect();
                let scale_vals: Vec<f32> = (0..NQ_BATCH)
                    .map(|i| {
                        let qi = if qi_start + i < qi_end { qi_start + i } else { pad_qi };
                        query_luts[qi].scale
                    })
                    .collect();
                let bias_vals: Vec<f32> = (0..NQ_BATCH)
                    .map(|i| {
                        let qi = if qi_start + i < qi_end { qi_start + i } else { pad_qi };
                        query_luts[qi].bias
                    })
                    .collect();

                // Per-query heaps
                let mut heap_scores: Vec<Vec<f32>> = (0..batch_nq)
                    .map(|_| vec![f32::NEG_INFINITY; k]).collect();
                let mut heap_indices: Vec<Vec<u32>> = (0..batch_nq)
                    .map(|_| vec![0u32; k]).collect();
                let mut heap_sizes = vec![0usize; batch_nq];
                let mut heap_mins = vec![f32::NEG_INFINITY; batch_nq];
                let mut heap_min_idxs = vec![0usize; batch_nq];

                unsafe {
                    if is_x86_feature_detected!("avx2") {
                        search_multi_query_avx2(
                            codes_slice, &lut_refs, &scale_vals, &bias_vals,
                            n_byte_groups, norms_slice, n_vectors,
                            batch_nq, k,
                            &mut heap_scores, &mut heap_indices,
                            &mut heap_sizes, &mut heap_mins, &mut heap_min_idxs,
                        );
                    }
                }

                // Sort each query's heap
                let mut batch_results = Vec::with_capacity(batch_nq);
                for qo in 0..batch_nq {
                    let sz = heap_sizes[qo];
                    let mut pairs: Vec<(f32, u32)> = heap_scores[qo][..sz].iter()
                        .zip(heap_indices[qo][..sz].iter())
                        .map(|(&s, &i)| (s, i)).collect();
                    pairs.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                    batch_results.push((
                        pairs.iter().map(|p| p.0).collect::<Vec<f32>>(),
                        pairs.iter().map(|p| p.1 as i64).collect::<Vec<i64>>(),
                    ));
                }
                batch_results
            })
            .collect();
        results
    };

    let mut top_scores = Array2::<f32>::zeros((nq, k));
    let mut top_indices = Array2::<i64>::zeros((nq, k));
    for (qi, (s, i)) in results.into_iter().enumerate() {
        for j in 0..s.len().min(k) {
            top_scores[[qi, j]] = s[j];
            top_indices[[qi, j]] = i[j];
        }
    }

    let t_score = t_score_start.elapsed().as_micros();
    eprintln!("search_inner({nq}q): rot={t_rot}us, lut={t_lut}us, score+heap={t_score}us, total={}us",
              t_rot + t_lut + t_score);

    (top_scores, top_indices)
}
// EVOLVE-BLOCK-END