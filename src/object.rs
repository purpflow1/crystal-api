use std::{cell::RefCell, sync::Arc};

use crate::{ObjectMemoryManager, Pipeline, Texture, mesh::Mesh};

pub struct Object {
    pub pipeline: Arc<dyn Pipeline>,
    pub mesh: Option<Arc<Mesh>>,
    pub(crate) memory_manager: Option<Box<dyn ObjectMemoryManager>>,
    pub textures: Option<Vec<(u32, Arc<dyn Texture>)>>,
}

#[allow(dead_code)]
impl Object {
    pub fn new(pipeline: Arc<dyn Pipeline>) -> Arc<RefCell<Self>> {
        Arc::new(RefCell::new(Self {
            pipeline,
            mesh: None,
            memory_manager: None,
            textures: None,
        }))
    }

    pub fn with_mesh(pipeline: Arc<dyn Pipeline>, mesh: Arc<Mesh>) -> Arc<RefCell<Self>> {
        Arc::new(RefCell::new(Self {
            pipeline,
            mesh: Some(mesh),
            memory_manager: None,
            textures: None,
        }))
    }

    pub fn with_mesh_textured(
        pipeline: Arc<dyn Pipeline>,
        mesh: Arc<Mesh>,
        textures: &[(u32, Arc<dyn Texture>)],
    ) -> Arc<RefCell<Self>> {
        Arc::new(RefCell::new(Self {
            pipeline,
            mesh: Some(mesh),
            memory_manager: None,
            textures: Some(textures.to_vec()),
        }))
    }
}
