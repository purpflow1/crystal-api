use std::{fs::File, path::Path};

use crate::errors::CrystalResult;

pub struct Image2D {
    pub width: u32,
    pub height: u32,
    pub channels: u32,
    pub pixels: Vec<u8>,
}

impl Image2D {
    pub fn new(path: &Path) -> CrystalResult<Self> {
        let file = File::open(path).unwrap();
        let decoder = png::Decoder::new(file);
        let mut reader = decoder.read_info().unwrap();
        let mut pixels = vec![0; reader.output_buffer_size()];
        let info = reader.next_frame(&mut pixels).unwrap();

        Ok(Self {
            width: info.width,
            height: info.height,
            channels: info.bit_depth as u32,
            pixels,
        })
    }
}
