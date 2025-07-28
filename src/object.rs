use std::sync::Arc;

use crate::{Buffer, GpuSamplerSet, Pipeline, mesh::Mesh};

/// Used for GPU mesh data
pub struct MeshBuffer {
    pub(crate) mesh: Arc<Mesh>,
    pub(crate) vertices: Arc<dyn Buffer>,
    pub(crate) indices: Arc<dyn Buffer>,
}

/// Unified object used in GPU operations
pub struct Object {
    pub(crate) pipeline: Arc<dyn Pipeline>,
    pub(crate) mesh_buffer: Option<Arc<MeshBuffer>>,
    pub(crate) sampler: Option<Arc<GpuSamplerSet>>,
    pub(crate) groups: Option<[u32; 3]>,
    pub(crate) array: u32,
}

unsafe impl Sync for Object {}
unsafe impl Send for Object {}

#[allow(dead_code)]
impl Object {
    /// Creates compute object
    pub fn new_compute(pipeline: Arc<dyn Pipeline>, groups: [u32; 3]) -> Arc<Self> {
        Arc::new(Self {
            pipeline,
            mesh_buffer: None,
            sampler: None,
            groups: Some(groups),
            array: 0,
        })
    }

    /// Creates compute object with textures
    pub fn compute_with_textures(
        pipeline: Arc<dyn Pipeline>,
        groups: [u32; 3],
        sampler: Arc<GpuSamplerSet>,
    ) -> Arc<Self> {
        Arc::new(Self {
            pipeline,
            mesh_buffer: None,
            sampler: Some(sampler),
            groups: Some(groups),
            array: 1,
        })
    }

    /// Creates graphics object with mesh only
    pub fn with_mesh(pipeline: Arc<dyn Pipeline>, mesh: Arc<MeshBuffer>) -> Arc<Self> {
        Arc::new(Self {
            pipeline,
            mesh_buffer: Some(mesh),
            sampler: None,
            groups: None,
            array: 1,
        })
    }

    /// Creates graphics object with mesh and textures
    pub fn with_mesh_sampled(
        pipeline: Arc<dyn Pipeline>,
        mesh: Arc<MeshBuffer>,
        sampler: Arc<GpuSamplerSet>,
    ) -> Arc<Self> {
        Arc::new(Self {
            pipeline,
            mesh_buffer: Some(mesh),
            sampler: Some(sampler),
            groups: None,
            array: 1,
        })
    }

    /// Creates array of graphics objects with mesh
    pub fn with_mesh_array(
        pipeline: Arc<dyn Pipeline>,
        mesh: Arc<MeshBuffer>,
        array: u32,
    ) -> Arc<Self> {
        Arc::new(Self {
            pipeline,
            mesh_buffer: Some(mesh),
            sampler: None,
            groups: None,
            array,
        })
    }

    /// Creates array of graphics objects with mesh and textures
    pub fn with_mesh_sampled_array(
        pipeline: Arc<dyn Pipeline>,
        mesh: Arc<MeshBuffer>,
        sampler: Arc<GpuSamplerSet>,
        array: u32,
    ) -> Arc<Self> {
        Arc::new(Self {
            pipeline,
            mesh_buffer: Some(mesh),
            sampler: Some(sampler),
            groups: None,
            array,
        })
    }
}
