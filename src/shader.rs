use std::{fs::File, io::Read};

use crate::{
    debug::log,
    errors::{CrystalError, CrystalResult},
};

#[derive(Debug)]
pub enum ShaderStage {
    Vertex,
    Fragment,
    Geometry,
    Tesselate,
    Compute,
}

pub struct Shader {
    pub(crate) stage: ShaderStage,
    pub(crate) code: Vec<u32>,
}

impl Shader {
    pub fn open(path: &str, stage: ShaderStage) -> CrystalResult<Self> {
        let mut file = match File::open(path) {
            Ok(file) => file,
            Err(e) => {
                log!("cannot open file: {}", e);
                return Err(CrystalError::ShaderError);
            }
        };

        let mut shader_code_bytes = Vec::<u8>::new();

        match file.read_to_end(&mut shader_code_bytes) {
            Ok(size) => unsafe {
                shader_code_bytes.set_len(size);
            },
            Err(e) => {
                log!("cannot read file: {}", e);
                return Err(CrystalError::ShaderError);
            }
        };

        let shader_code = unsafe {
            let len = shader_code_bytes.len();

            let ptr = std::alloc::alloc(std::alloc::Layout::from_size_align_unchecked(len, 0x10))
                as *mut u8;

            if ptr.is_null() {
                panic!("Failed to allocate memory");
            }

            std::slice::from_raw_parts_mut(ptr, len).copy_from_slice(std::slice::from_raw_parts(
                shader_code_bytes.as_ptr(),
                shader_code_bytes.len(),
            ));

            Vec::from_raw_parts(ptr as *mut u32, len / 4, len / 4)
        };

        Ok(Shader {
            stage,
            code: shader_code,
        })
    }
}
