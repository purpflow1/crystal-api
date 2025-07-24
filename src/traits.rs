use std::{ops::Range, sync::Arc};

use crate::{
    GpuSamplerSet,
    errors::GraphicsResult,
    mesh::{Attribute, Mesh},
    object::{MeshBuffer, Object},
    shader::Shader,
    vulkan,
};

/// ```GraphicsApi``` is the main trait used for creation of graphics resources and using this wrapper
pub trait GraphicsApi: Sync + Send {
    /// Executes all operations permitted with objects passed in this method
    fn dispatch_any(&self, objects: &[Arc<Object>]) -> GraphicsResult<()>;
    /// Executes all compute operations permitted with objects passed in this method
    fn dispatch_compute(&self, objects: &[Arc<Object>]) -> GraphicsResult<()>;

    /// Resizes resources
    fn resize_resources(&self, width: u32, height: u32) -> GraphicsResult<()>;

    /// Returns ```RenderTarget``` created on presentation init. Panics if API was initialized without presentation
    fn get_presentation_render_target(&self) -> Arc<dyn RenderTarget>;
    /// Creates shader layout
    fn create_layout(
        &self,
        double_buffering: bool,
        texture_num: usize,
        sampler_num: usize,
        uniform_num: usize,
        storage_num: usize,
    ) -> GraphicsResult<Arc<dyn Layout>>;
    /// Creates GPU buffer
    fn create_buffer(
        &self,
        size: u64,
        uniform: bool,
        transfer: bool,
        enable_sync: bool,
    ) -> GraphicsResult<Arc<dyn Buffer>>;
    /// Creates a set of GPU buffers used for meshes
    fn create_buffer_mesh(&self, mesh: Arc<Mesh>) -> GraphicsResult<Arc<MeshBuffer>>;
    /// Creates sampler set
    fn create_sampler_set(
        &self,
        textures: &[(u32, Arc<dyn Texture>)],
    ) -> GraphicsResult<Arc<GpuSamplerSet>>;
    /// Creates texture
    fn create_texture(
        &self,
        buffer: Arc<dyn Buffer>,
        data: [u32; 3],
        anisotropy_texels: f32,
    ) -> GraphicsResult<Arc<dyn Texture>>;

    /// Returns the duration of previous frame
    fn get_delta_time(&self) -> std::time::Duration;
}

/// Contains resources used to rendering in them
pub trait RenderTarget: Sync + Send {
    #[allow(missing_docs)]
    fn as_vulkan(self: Arc<Self>) -> Option<Arc<vulkan::VulkanRenderTarget>> {
        None
    }
}

/// Contains resources used to mapping GPU memory in shaders
pub trait Layout: Sync + Send {
    #[allow(missing_docs)]
    fn as_vulkan(self: Arc<Self>) -> Option<Arc<vulkan::VulkanLayout>> {
        None
    }

    /// Creates graphics pipeline
    fn create_graphics_pipeline(
        self: Arc<Self>,
        render_target: Arc<dyn RenderTarget>,
        shaders: &[Shader],
        attributes: &[Attribute],
    ) -> GraphicsResult<Arc<dyn Pipeline>>;

    /// Creates compute pipeline
    fn create_compute_pipeline(
        self: Arc<Self>,
        shader: &Shader,
    ) -> GraphicsResult<Arc<dyn Pipeline>>;

    /// Registers samplers in layout for reusing them.
    /// Samplers are removing automatically on zero hard references in ```Arc```
    fn register_samplers(&self, samplers: &[Arc<GpuSamplerSet>]) -> GraphicsResult<()>;
    /// Adds buffer
    fn add_buffer(&self, binding: u32, buffer: Arc<dyn Buffer>) -> GraphicsResult<()>;
}

/// Used to store data about texture in GPU
pub trait Texture: Sync + Send {
    #[allow(missing_docs)]
    fn as_vulkan(self: Arc<Self>) -> Option<Arc<vulkan::VulkanTexture>> {
        None
    }
}

/// Contains compiled shader stages and layout bindings info
pub trait Pipeline: Sync + Send {
    #[allow(missing_docs)]
    fn as_vulkan(self: Arc<Self>) -> Option<Arc<vulkan::VulkanPipeline>> {
        None
    }
}

/// Used to store GPU data
pub trait Buffer: Sync + Send {
    #[allow(missing_docs)]
    fn as_vulkan(self: Arc<Self>) -> Option<Arc<vulkan::BufferManager>> {
        None
    }

    /// Returns a slice of this buffer in given range
    fn get_memory(&self, range: Range<usize>) -> &mut [u8];
    /// Returns a slice of this buffer
    fn get_memory_full(&self) -> &mut [u8];
}
