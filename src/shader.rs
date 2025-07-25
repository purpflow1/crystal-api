use crate::errors::GraphicsResult;

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
    /// Creates new shader from words and specified stage
    /// ```rust
    /// let shader = Shader::from_words(words, ShaderStage::Vertex).unwrap()
    /// ```
    pub fn from_words(words: &[u32], stage: ShaderStage) -> GraphicsResult<Self> {
        Ok(Shader {
            stage,
            code: words.to_vec(),
        })
    }

    /// Creates new shader from bytes and specified stage
    /// ```rust
    /// let shader = Shader::from_bytes(bytes, ShaderStage::Vertex).unwrap()
    /// ```
    pub fn from_bytes(bytes: &[u8], stage: ShaderStage) -> GraphicsResult<Self> {
        let shader_code = unsafe {
            let ptr = std::alloc::alloc(std::alloc::Layout::from_size_align_unchecked(
                bytes.len(),
                0x10,
            )) as *mut u8;

            if ptr.is_null() {
                panic!("Failed to allocate memory");
            }

            std::slice::from_raw_parts_mut(ptr, bytes.len())
                .copy_from_slice(std::slice::from_raw_parts(bytes.as_ptr(), bytes.len()));

            let len = bytes.len() / 4;

            Vec::from_raw_parts(ptr as *mut u32, len, len)
        };

        Ok(Shader {
            stage,
            code: shader_code,
        })
    }
}
