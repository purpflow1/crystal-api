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
    pub mesh_buffer: Arc<MeshBuffer>,
    pub samplers: Option<Vec<(u32, Arc<GpuSampler>)>>,
}

unsafe impl Sync for Object {}
unsafe impl Send for Object {}

#[allow(dead_code)]
impl Object {
    pub fn with_mesh(pipeline: Arc<dyn Pipeline>, mesh: Arc<MeshBuffer>) -> Arc<Self> {
        Arc::new(Self {
            id: Mutex::new(usize::MAX),
            pipeline,
            mesh_buffer: mesh,
            samplers: None,
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
            mesh_buffer: mesh,
            samplers: Some(textures.to_vec()),
        })
    }
}
