use std::sync::Arc;

use crate::{Buffer, GpuSamplerSet, Pipeline, mesh::Mesh};

pub struct MeshBuffer {
    pub mesh: Arc<Mesh>,
    pub vertices: Arc<dyn Buffer>,
    pub indices: Arc<dyn Buffer>,
}

pub struct Object {
    pub pipeline: Arc<dyn Pipeline>,
    pub mesh_buffer: Option<Arc<MeshBuffer>>,
    pub sampler: Option<Arc<GpuSamplerSet>>,
    pub groups: Option<[u32; 3]>,
    pub array: u32,
}

unsafe impl Sync for Object {}
unsafe impl Send for Object {}

#[allow(dead_code)]
impl Object {
    pub fn new_compute(pipeline: Arc<dyn Pipeline>, groups: [u32; 3]) -> Arc<Self> {
        Arc::new(Self {
            pipeline,
            mesh_buffer: None,
            sampler: None,
            groups: Some(groups),
            array: 0,
        })
    }

    pub fn with_mesh(pipeline: Arc<dyn Pipeline>, mesh: Arc<MeshBuffer>) -> Arc<Self> {
        Arc::new(Self {
            pipeline,
            mesh_buffer: Some(mesh),
            sampler: None,
            groups: None,
            array: 1,
        })
    }

    pub fn with_textures(pipeline: Arc<dyn Pipeline>, sampler: Arc<GpuSamplerSet>) -> Arc<Self> {
        Arc::new(Self {
            pipeline,
            mesh_buffer: None,
            sampler: Some(sampler),
            groups: None,
            array: 1,
        })
    }

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
