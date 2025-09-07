use std::sync::{Arc, Mutex};

use crate::traits;

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
