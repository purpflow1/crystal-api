use std::sync::Arc;

use crate::GpuSamplerSet;
use crate::proxies::*;

/// Used for GPU mesh data
pub(crate) struct MeshBuffer {
    pub(crate) index_size: usize,
    pub(crate) vertices: Arc<dyn BufferProxy>,
    pub(crate) indices: Arc<dyn BufferProxy>,
}

/// Unified object used in GPU operations
pub struct Object {
    pub(crate) pipeline: Arc<dyn PipelineProxy>,
    pub(crate) mesh_buffer: Option<Arc<MeshBuffer>>,
    pub(crate) sampler: Option<Arc<GpuSamplerSet>>,
    pub(crate) groups: Option<[u32; 3]>,
    pub(crate) index: u32,
    pub(crate) array: u32,
}

unsafe impl Sync for Object {}
unsafe impl Send for Object {}
