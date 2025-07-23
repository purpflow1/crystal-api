use std::{ops::Range, sync::Arc};

use crate::{GpuSamplerSet, errors::GraphicsResult, mesh::Attribute, shader::Shader, vulkan};

pub trait RenderTarget: Sync + Send {
    fn as_vulkan(self: Arc<Self>) -> Option<Arc<vulkan::VulkanRenderTarget>> {
        None
    }
}

pub trait Layout: Sync + Send {
    fn as_vulkan(self: Arc<Self>) -> Option<Arc<vulkan::VulkanLayout>> {
        None
    }

    fn create_graphics_pipeline(
        self: Arc<Self>,
        render_target: Arc<dyn RenderTarget>,
        shaders: &[Shader],
        attributes: &[Attribute],
    ) -> GraphicsResult<Arc<dyn Pipeline>>;

    fn create_compute_pipeline(
        self: Arc<Self>,
        shader: &Shader,
    ) -> GraphicsResult<Arc<dyn Pipeline>>;

    fn register_samplers(&self, samplers: &[Arc<GpuSamplerSet>]) -> GraphicsResult<()>;
    fn add_buffer(&self, binding: u32, buffer: Arc<dyn Buffer>) -> GraphicsResult<()>;
}

pub trait Texture: Sync + Send {
    fn as_vulkan(self: Arc<Self>) -> Option<Arc<vulkan::VulkanTexture>> {
        None
    }
}

pub trait Pipeline: Sync + Send {
    fn as_vulkan(self: Arc<Self>) -> Option<Arc<vulkan::VulkanPipeline>> {
        None
    }
}

pub trait Buffer: Sync + Send {
    fn as_vulkan(self: Arc<Self>) -> Option<Arc<vulkan::BufferManager>> {
        None
    }

    fn get_memory(&self, range: Range<usize>) -> &mut [u8];
    fn get_memory_full(&self) -> &mut [u8];
}
