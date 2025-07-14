use std::{
    sync::{Arc, Mutex},
    usize,
};

use crate::{Buffer, GpuSampler, Pipeline, mesh::Mesh};

pub struct MeshBuffer {
    pub mesh: Arc<Mesh>,
    pub vertices: Arc<dyn Buffer>,
    pub indices: Arc<dyn Buffer>,
}

pub struct Object {
    pub(crate) id: Mutex<usize>,
    pub pipeline: Arc<dyn Pipeline>,
    pub mesh_buffer: Option<Arc<MeshBuffer>>,
    pub samplers: Option<Vec<(u32, Arc<GpuSampler>)>>,
    pub groups: Option<[u32; 3]>,
}

unsafe impl Sync for Object {}
unsafe impl Send for Object {}

#[allow(dead_code)]
impl Object {
    pub fn new_compute(pipeline: Arc<dyn Pipeline>, groups: [u32; 3]) -> Arc<Self> {
        Arc::new(Self {
            id: Mutex::new(usize::MAX),
            pipeline,
            mesh_buffer: None,
            samplers: None,
            groups: Some(groups),
        })
    }

    pub fn with_mesh(pipeline: Arc<dyn Pipeline>, mesh: Arc<MeshBuffer>) -> Arc<Self> {
        Arc::new(Self {
            id: Mutex::new(usize::MAX),
            pipeline,
            mesh_buffer: Some(mesh),
            samplers: None,
            groups: None,
        })
    }

    pub fn with_textures(
        pipeline: Arc<dyn Pipeline>,
        textures: &[(u32, Arc<GpuSampler>)],
    ) -> Arc<Self> {
        Arc::new(Self {
            id: Mutex::new(usize::MAX),
            pipeline,
            mesh_buffer: None,
            samplers: Some(textures.to_vec()),
            groups: None,
        })
    }

    pub fn with_mesh_textured(
        pipeline: Arc<dyn Pipeline>,
        mesh: Arc<MeshBuffer>,
        textures: &[(u32, Arc<GpuSampler>)],
    ) -> Arc<Self> {
        Arc::new(Self {
            id: Mutex::new(usize::MAX),
            pipeline,
            mesh_buffer: Some(mesh),
            samplers: Some(textures.to_vec()),
            groups: None,
        })
    }
}
