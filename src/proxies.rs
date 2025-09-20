use std::{ops::Range, sync::Arc};

use crate::{
    GpuSamplerSet,
    errors::GraphicsResult,
    mesh::Attribute,
    object::{MeshBufferProxy, Object},
    shader::Shader,
    vulkan,
};

pub(crate) trait DeviceProxy: Sync + Send {
    fn dispatch_and_present(&self, objects: &[Arc<Object>]) -> GraphicsResult<()>;
    fn dispatch_compute(&self, objects: &[Arc<Object>]) -> GraphicsResult<()>;
    fn resize_resources(&self, width: u32, height: u32) -> GraphicsResult<()>;
    fn get_presentation_render_target(&self) -> Option<Arc<dyn RenderTargetProxy>>;
    fn create_layout(
        &self,
        double_buffering: bool,
        texture_num: usize,
        sampler_num: usize,
        uniform_num: usize,
        storage_num: usize,
    ) -> GraphicsResult<Arc<dyn LayoutProxy>>;
    fn create_buffer(
        &self,
        size: u64,
        uniform: bool,
        transfer: bool,
        enable_sync: bool,
    ) -> GraphicsResult<Arc<dyn BufferProxy>>;
    fn create_buffer_mesh(
        &self,
        vertices: &[u8],
        indices: &[u8],
        index_size: usize,
    ) -> GraphicsResult<Arc<MeshBufferProxy>>;
    fn create_sampler_set(
        &self,
        textures: &[(u32, Arc<dyn TextureProxy>)],
        layouts: &[Arc<dyn LayoutProxy>],
    ) -> GraphicsResult<Arc<GpuSamplerSet>>;
    fn create_texture(
        &self,
        buffer: Arc<dyn BufferProxy>,
        extent: [u32; 2],
        anisotropy_texels: f32,
    ) -> GraphicsResult<Arc<dyn TextureProxy>>;
    fn get_delta_time(&self) -> std::time::Duration;
}

pub(crate) trait RenderTargetProxy: Sync + Send {
    fn as_vulkan(self: Arc<Self>) -> Option<Arc<vulkan::VulkanRenderTarget>> {
        None
    }
    fn create_render_target(
        &self,
        extent: [u32; 2],
        anisotropy_texels: f32,
        msaa_samples: u8,
    ) -> GraphicsResult<(Arc<dyn RenderTargetProxy>, Arc<dyn TextureProxy>)>;
}

pub(crate) trait LayoutProxy: Sync + Send {
    fn create_graphics_pipeline(
        self: Arc<Self>,
        render_target: Arc<dyn RenderTargetProxy>,
        shaders: &[Shader],
        attributes: &[Attribute],
    ) -> GraphicsResult<Arc<dyn PipelineProxy>>;
    fn create_compute_pipeline(
        self: Arc<Self>,
        shader: &Shader,
    ) -> GraphicsResult<Arc<dyn PipelineProxy>>;
    fn register_samplers(&self, samplers: &[Arc<GpuSamplerSet>]) -> GraphicsResult<()>;
    fn add_buffer(&self, binding: u32, buffer: Arc<dyn BufferProxy>) -> GraphicsResult<()>;
}

pub(crate) trait TextureProxy: Sync + Send {
    fn as_vulkan(self: Arc<Self>) -> Option<Arc<vulkan::VulkanTexture>> {
        None
    }
}

pub(crate) trait PipelineProxy: Sync + Send {
    fn as_vulkan(self: Arc<Self>) -> Option<Arc<vulkan::VulkanPipeline>> {
        None
    }
}

pub(crate) trait BufferProxy: Sync + Send {
    fn as_vulkan(self: Arc<Self>) -> Option<Arc<vulkan::BufferManager>> {
        None
    }

    fn get_memory<'a>(&self, range: Range<u64>) -> &'a mut [u8];
    fn get_size(&self) -> u64;
}
