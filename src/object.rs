use std::{
    sync::{Arc, Mutex, RwLock},
    usize,
};

use crate::{GpuSampler, ObjectMemoryManager, Pipeline, mesh::Mesh};

pub struct Object {
    pub(crate) id: Mutex<usize>,
    pub pipeline: Arc<dyn Pipeline>,
    pub mesh: Option<Arc<Mesh>>,
    pub(crate) memory_manager: RwLock<Option<Box<dyn ObjectMemoryManager>>>,
    pub samplers: Option<Vec<(u32, Arc<GpuSampler>)>>,
}

unsafe impl Sync for Object {}
unsafe impl Send for Object {}

#[allow(dead_code)]
impl Object {
    pub fn new(pipeline: Arc<dyn Pipeline>) -> Arc<Self> {
        Arc::new(Self {
            id: Mutex::new(usize::MAX),
            pipeline,
            mesh: None,
            memory_manager: RwLock::new(None),
            samplers: None,
        })
    }

    pub fn with_mesh(pipeline: Arc<dyn Pipeline>, mesh: Arc<Mesh>) -> Arc<Self> {
        Arc::new(Self {
            id: Mutex::new(usize::MAX),
            pipeline,
            mesh: Some(mesh),
            memory_manager: RwLock::new(None),
            samplers: None,
        })
    }

    pub fn with_mesh_textured(
        pipeline: Arc<dyn Pipeline>,
        mesh: Arc<Mesh>,
        textures: &[(u32, Arc<GpuSampler>)],
    ) -> Arc<Self> {
        Arc::new(Self {
            id: Mutex::new(usize::MAX),
            pipeline,
            mesh: Some(mesh),
            memory_manager: RwLock::new(None),
            samplers: Some(textures.to_vec()),
        })
    }
}
