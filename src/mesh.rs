use std::{marker::PhantomData, sync::Arc};

/// Used to setup vertex shader attributes
#[derive(Clone, Copy)]
pub struct Attribute {
    /// Size of the attribute
    pub size: usize,
    /// Offset of the attribute
    pub offset: usize,
}

/// Needed to be implemented for Vertex for using it in Mesh
///
/// # Examples
/// ```rust
/// use std::mem::offset_of;
/// use crystal_api::mesh::{AttributeDescriptor, Attribute};
///
/// type Vec3 = [f32; 3];
/// type Vec2 = [f32; 2];
///
/// #[repr(C)]
/// #[derive(Clone, Copy, Default)]
/// pub struct VertexTexture {
///     pos: Vec3,
///     uv: Vec2,
/// }
/// impl AttributeDescriptor for VertexTexture {
///     fn get_attributes() -> &'static [Attribute] {
///         &[
///             Attribute {
///                 size: size_of::<Vec3>(),
///                 offset: offset_of!(Self, pos),
///             },
///             Attribute {
///                 size: size_of::<Vec2>(),
///                 offset: offset_of!(Self, uv),
///             },
///         ]
///     }
/// }
/// ```
pub trait AttributeDescriptor {
    /// Should return attributes of Vertex
    fn get_attributes() -> &'static [Attribute];
}

/// `Mesh` struct stores the mesh data in CPU memory
pub struct Mesh<V: AttributeDescriptor, I> {
    #[allow(missing_docs)]
    pub vertices: Vec<V>,
    #[allow(missing_docs)]
    pub indices: Vec<I>,
}

/// `MeshBuffer` struct stores the mesh data in GPU memory
pub struct MeshBuffer<V: AttributeDescriptor, I> {
    pub(crate) inner: Arc<crate::object::MeshBufferProxy>,
    _tp: PhantomData<(V, I)>,
}

impl<V: AttributeDescriptor, I> MeshBuffer<V, I> {
    pub(crate) fn new(inner: Arc<crate::object::MeshBufferProxy>) -> Self {
        Self {
            inner,
            _tp: PhantomData::default(),
        }
    }
}
