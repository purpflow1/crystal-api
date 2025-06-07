use std::{fs::File, io::Read};

use crate::{
    GpuVec,
    debug::log,
    errors::{CrystalError, CrystalResult},
};

pub enum ShaderStage {
    Vertex,
    Fragment,
    Geometry,
}

pub struct Shader {
    pub(crate) stage: ShaderStage,
    pub(crate) code: GpuVec,
}

impl Shader {
    pub fn open(path: &str, stage: ShaderStage) -> CrystalResult<Self> {
        let mut shader_code_bytes = Vec::<u8>::new();

        let mut file = match File::open(path) {
            Ok(file) => file,
            Err(e) => {
                log!("cannot open file: {}", e);
                return Err(CrystalError::ShaderError);
            }
        };

        match file.read_to_end(&mut shader_code_bytes) {
            Ok(size) => unsafe {
                shader_code_bytes.set_len(size);
            },
            Err(e) => {
                log!("cannot read file: {}", e);
                return Err(CrystalError::ShaderError);
            }
        };

        Ok(Shader {
            stage,
            code: GpuVec::new(&shader_code_bytes),
        })
    }
}
