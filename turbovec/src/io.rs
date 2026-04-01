//! Read/write TurboVec index files (.tv format).
//!
//! Binary format: 9-byte header + packed codes + norms
//!   Header: bit_width (u8) | dim (u32 LE) | n_vectors (u32 LE)

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

const HEADER_SIZE: usize = 9;

pub fn write(
    path: impl AsRef<Path>,
    bit_width: usize,
    dim: usize,
    n_vectors: usize,
    packed_codes: &[u8],
    norms: &[f32],
) -> io::Result<()> {
    let mut f = BufWriter::new(File::create(path)?);

    // Header
    f.write_all(&[bit_width as u8])?;
    f.write_all(&(dim as u32).to_le_bytes())?;
    f.write_all(&(n_vectors as u32).to_le_bytes())?;

    // Packed codes
    f.write_all(packed_codes)?;

    // Norms as raw f32 bytes
    for &n in norms {
        f.write_all(&n.to_le_bytes())?;
    }

    f.flush()?;
    Ok(())
}

pub fn load(path: impl AsRef<Path>) -> io::Result<(usize, usize, usize, Vec<u8>, Vec<f32>)> {
    let mut f = BufReader::new(File::open(path)?);

    // Header
    let mut header = [0u8; HEADER_SIZE];
    f.read_exact(&mut header)?;

    let bit_width = header[0] as usize;
    let dim = u32::from_le_bytes([header[1], header[2], header[3], header[4]]) as usize;
    let n_vectors = u32::from_le_bytes([header[5], header[6], header[7], header[8]]) as usize;

    // Packed codes
    let packed_bytes = (dim / 8) * bit_width * n_vectors;
    let mut packed_codes = vec![0u8; packed_bytes];
    f.read_exact(&mut packed_codes)?;

    // Norms
    let mut norms_bytes = vec![0u8; n_vectors * 4];
    f.read_exact(&mut norms_bytes)?;
    let norms: Vec<f32> = norms_bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    Ok((bit_width, dim, n_vectors, packed_codes, norms))
}
