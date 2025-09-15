use std::sync::Arc;

use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

use crate::{
    GpuSamplerSet, GraphicsApiInitSettings,
    buffer::Buffer,
    errors::GraphicsResult,
    layout::Layout,
    mesh::{AttributeDescriptor, Mesh},
    object::{MeshBuffer, Object},
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
        Ok(Self {
            inner: VulkanEntry::no_presentation()?,
        })
    }

    /// Initializes device with presentation support
    pub fn graphics<T: HasWindowHandle + HasDisplayHandle>(
        settings: &GraphicsApiInitSettings,
        window: &T,
    ) -> GraphicsResult<Self> {
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
        if let Some(render_target) = self.inner.get_presentation_render_target() {
            Some(RenderTarget::new(render_target))
        } else {
            None
        }
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
        Ok(Layout::new(self.inner.create_layout(
            double_buffering,
            texture_num,
            sampler_num,
            uniform_num,
            storage_num,
        )?))
    }

    /// Creates GPU buffer
    pub fn create_buffer<T>(
        &self,
        len: u64,
        uniform: bool,
        transfer: bool,
        enable_sync: bool,
    ) -> GraphicsResult<Buffer<T>> {
        Ok(Buffer::new(self.inner.create_buffer(
            len * size_of::<T>() as u64,
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
