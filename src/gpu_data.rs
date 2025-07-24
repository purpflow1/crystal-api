use std::{
    sync::{Arc, Mutex},
    usize,
};

use crate::traits;

/// ```AsBytes``` represents a memory region as slice of bytes
pub trait AsBytes {
    /// ```as_bytes``` represents a memory region as slice of bytes
    fn as_bytes(&self) -> &[u8];
}

impl<T> AsBytes for Vec<T> {
    fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(self.as_ptr() as *const u8, self.len() * size_of::<T>())
        }
    }
}

impl<T> AsBytes for &[T] {
    fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(self.as_ptr() as *const u8, self.len() * size_of::<T>())
        }
    }
}

/// ## Used for textures bindings in shaders
pub struct GpuSamplerSet {
    pub(crate) id: Mutex<usize>,
    pub(crate) textures: Vec<(u32, Arc<dyn traits::Texture>)>,
}

impl GpuSamplerSet {
    /// Creates new GpuSamplerSet from textures and its bindings
    pub fn from_textures(textures: &[(u32, Arc<dyn traits::Texture>)]) -> Arc<Self> {
        Arc::new(Self {
            id: Mutex::new(usize::MAX),
            textures: textures.to_vec(),
        })
    }
}
