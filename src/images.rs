use image::ImageReader;
use std::path::Path;

use crate::errors::CrystalResult;

pub struct Image2D {
    pub width: u32,
    pub height: u32,
    pub channels: u32,
    pub pixels: Vec<u8>,
}

impl Image2D {
    pub fn new(path: &Path) -> CrystalResult<Self> {
        let img = ImageReader::open(path)
            .unwrap()
            .decode()
            .unwrap()
            .into_rgba8();

        Ok(Self {
            width: img.width(),
            height: img.height(),
            channels: 4,
            pixels: img.into_vec(),
        })
    }
}
