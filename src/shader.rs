use std::{fs::File, io::Read};

use crate::{
    debug::log,
    errors::{GraphicsError, GraphicsResult},
};

#[allow(missing_docs)]
#[derive(Debug)]
pub enum ShaderStage {
    Vertex,
    Fragment,
    Geometry,
    Tesselate,
    Compute,
}

#[allow(missing_docs)]
pub struct Shader {
    pub(crate) stage: ShaderStage,
    pub(crate) code: Vec<u32>,
}

impl Shader {
    /// Opens shader with path and specified stage
    /// ```rust
    /// let shader = Shader::open(
    ///    "shader.vert.spv",
    ///    ShaderStage::Vertex,
    ///).unwrap()
    /// ```
    pub fn open(path: &str, stage: ShaderStage) -> GraphicsResult<Self> {
        let mut file = match File::open(path) {
            Ok(file) => file,
            Err(e) => {
                log!("cannot open file: {}", e);
                return Err(GraphicsError::ShaderError);
            }
        };

        let size = file.metadata().unwrap().len() as usize;

        let mut shader_code_bytes = Vec::<u8>::with_capacity(size);

        match file.read_to_end(&mut shader_code_bytes) {
            Ok(_) => unsafe {
                shader_code_bytes.set_len(size);
            },
            Err(e) => {
                log!("cannot read file: {}", e);
                return Err(GraphicsError::ShaderError);
            }
        };

        let shader_code = unsafe {
            let ptr = std::alloc::alloc(std::alloc::Layout::from_size_align_unchecked(size, 0x10))
                as *mut u8;

            if ptr.is_null() {
                panic!("Failed to allocate memory");
            }

            std::slice::from_raw_parts_mut(ptr, size).copy_from_slice(std::slice::from_raw_parts(
                shader_code_bytes.as_ptr(),
                shader_code_bytes.len(),
            ));

            let len = size / 4;

            Vec::from_raw_parts(ptr as *mut u32, len, len)
        };

        Ok(Shader {
            stage,
            code: shader_code,
        })
    }
}
