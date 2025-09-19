use std::sync::Arc;

use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

use crate::{
    GpuSamplerSet, GraphicsApiInitSettings,
    bitflags::BufferFlags,
    buffer::Buffer,
    debug::{error, fmt_size, log},
    errors::{GraphicsError, GraphicsResult},
    layout::Layout,
    mesh::{AttributeDescriptor, Mesh},
    object::Object,
    proxies::*,
    render_target::RenderTarget,
    texture::Texture,
    vulkan::VulkanEntry,
};

/// ```Device``` is the main struct used for creation of graphics resources and using this wrapper
pub struct Device {
    inner: Box<dyn DeviceProxy>,
}

impl Device {
    /// Initializes device with no presentation support
    pub fn compute() -> GraphicsResult<Self> {
        log!("creating compute device");
        Ok(Self {
            inner: VulkanEntry::no_presentation()?,
        })
    }

    /// Initializes device with presentation support
    pub fn graphics<T: HasWindowHandle + HasDisplayHandle>(
        settings: &GraphicsApiInitSettings,
        window: &T,
    ) -> GraphicsResult<Self> {
        log!("creating graphics device");
        Ok(Self {
            inner: VulkanEntry::with_presentation(settings, window)?,
        })
    }

    /// Executes all operations permitted with objects passed in this method
    pub fn dispatch_and_present(&self, objects: &[Arc<Object>]) -> GraphicsResult<()> {
        self.inner.dispatch_and_present(objects)
    }

    /// Executes all compute operations permitted with objects passed in this method
    pub fn dispatch_compute(&self, objects: &[Arc<Object>]) -> GraphicsResult<()> {
        self.inner.dispatch_compute(objects)
    }

    /// Resizes resources
    pub fn resize_resources(&self, width: u32, height: u32) -> GraphicsResult<()> {
        self.inner.resize_resources(width, height)
    }

    /// Returns ```RenderTarget``` created on presentation init
    pub fn get_presentation_render_target(&self) -> Option<RenderTarget> {
        self.inner
            .get_presentation_render_target()
            .map(RenderTarget::new)
    }

    /// Creates shader layout
    pub fn create_layout(
        &self,
        double_buffering: bool,
        texture_num: usize,
        sampler_num: usize,
        uniform_num: usize,
        storage_num: usize,
    ) -> GraphicsResult<Layout> {
        if (sampler_num == 0 && texture_num == 0) == (sampler_num > 0 && texture_num > 0) {
            error!("textures cannot exist without samplers");
            return Err(GraphicsError::DataError);
        }

        if !(uniform_num > 0 || storage_num > 0) {
            error!("cannot create layout without buffers (WARN vulkan specific, todo fix)");
            return Err(GraphicsError::DataError);
        } // TODO it is possible! Vulkan specific

        log!("creating layout [ double_buffering: {} ]", double_buffering);

        Ok(Layout::new(self.inner.create_layout(
            double_buffering,
            texture_num,
            sampler_num,
            uniform_num,
            storage_num,
        )?))
    }

    /// Creates GPU buffer
    pub fn create_buffer<T>(&self, len: u64, flags: BufferFlags) -> GraphicsResult<Buffer<T>> {
        let size = len * size_of::<T>() as u64;

        log!(
            "creating buffer [ size: {} bits: {:?} ]",
            fmt_size!(size),
            flags
        );

        let uniform = !(flags & BufferFlags::UNIFORM).is_none();
        let transfer = !(flags & BufferFlags::TRANSFER).is_none();
        let enable_sync = !(flags & BufferFlags::SYNCED).is_none();

        Ok(Buffer::new(self.inner.create_buffer(
            size,
            uniform,
            transfer,
            enable_sync,
        )?))
    }

    /// Creates a set of GPU buffers used for meshes
    pub fn create_buffer_mesh<V: AttributeDescriptor, I>(
        &self,
        mesh: &Mesh<V, I>,
    ) -> GraphicsResult<crate::mesh::MeshBuffer<V, I>> {
        log!("creating mesh [ size: {} ] ", {
            let size = mesh.size();
            fmt_size!(size)
        });

        let vertices = unsafe {
            std::slice::from_raw_parts(
                mesh.vertices.as_ptr() as *const u8,
                mesh.vertices.len() * size_of::<V>(),
            )
        };
        let indices = unsafe {
            std::slice::from_raw_parts(
                mesh.indices.as_ptr() as *const u8,
                mesh.indices.len() * size_of::<I>(),
            )
        };
        let buffer_mesh = self
            .inner
            .create_buffer_mesh(vertices, indices, size_of::<I>())?;

        Ok(crate::mesh::MeshBuffer::new(buffer_mesh))
    }

    /// Creates sampler set
    pub fn create_sampler_set(
        &self,
        textures: &[(u32, &Texture)],
        layouts: &[&Layout],
    ) -> GraphicsResult<Arc<GpuSamplerSet>> {
        log!(
            "creating sampler [ bindings: {:?} ]",
            textures
                .iter()
                .map(|(binding, _)| *binding)
                .collect::<Vec<_>>()
        );

        self.inner.create_sampler_set(
            &textures
                .iter()
                .map(|(b, texture)| (*b, texture.inner.clone()))
                .collect::<Vec<(u32, Arc<dyn TextureProxy>)>>(),
            &layouts
                .iter()
                .map(|layout| layout.inner.clone())
                .collect::<Vec<Arc<dyn LayoutProxy>>>(),
        )
    }

    /// Creates texture from buffers
    pub fn create_texture(
        &self,
        buffer: &Buffer<u8>,
        extent: [u32; 2],
        anisotropy_texels: f32,
    ) -> GraphicsResult<Texture> {
        let size = buffer.len();
        log!(
            "creating texture [ size: {} extent: {}x{} ]",
            fmt_size!(size),
            extent[0],
            extent[1]
        );
        Ok(Texture::new(self.inner.create_texture(
            buffer.inner.clone(),
            extent,
            anisotropy_texels,
        )?))
    }

    /// Returns the duration of previous frame
    pub fn get_delta_time(&self) -> std::time::Duration {
        self.inner.get_delta_time()
    }
}
