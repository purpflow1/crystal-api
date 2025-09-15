use std::{marker::PhantomData, sync::Arc};

/// Used to setup vertex shader attributes
#[derive(Clone, Copy)]
pub struct Attribute {
    pub size: usize,
    pub offset: usize,
}

pub trait AttributeDescriptor {
    fn get_attributes() -> &'static [Attribute];
}

/// Mesh struct stores the mesh data in CPU memory
pub struct Mesh<V: AttributeDescriptor, I> {
    pub vertices: Vec<V>,
    pub indices: Vec<I>,
}

pub struct MeshBuffer<V: AttributeDescriptor, I> {
    pub(crate) inner: Arc<crate::object::MeshBuffer>,
    _tp: PhantomData<(V, I)>,
}

impl<V: AttributeDescriptor, I> MeshBuffer<V, I> {
    pub(crate) fn new(inner: Arc<crate::object::MeshBuffer>) -> Self {
        Self {
            inner,
            _tp: PhantomData::default(),
        }
    }
}
