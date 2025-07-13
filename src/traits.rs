use std::sync::{Arc, Mutex};

use crate::{errors::CrystalResult, mesh::Attribute, object::Object, shader::Shader, vulkan};

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
    ) -> CrystalResult<Arc<dyn Pipeline>>;

    fn create_compute_pipeline(
        self: Arc<Self>,
        shader: &Shader,
    ) -> CrystalResult<Arc<dyn Pipeline>>;

    fn register_samplers(&self, objects: &[Arc<Object>]) -> CrystalResult<()>;
    fn add_buffer(&self, binding: u32, buffer: Arc<dyn Buffer>) -> CrystalResult<()>;
}

pub trait Texture: Sync + Send {
    fn as_vulkan(self: Arc<Self>) -> Option<Arc<vulkan::VulkanTexture>> {
        None
    }
}

pub(crate) trait ObjectMemoryManager: Sync + Send {
    fn as_vulkan_mut(&mut self) -> Option<&mut vulkan::VulkanObjectMemoryManager> {
        None
    }

    fn as_vulkan_ref(&self) -> Option<&vulkan::VulkanObjectMemoryManager> {
        None
    }
}

pub trait Pipeline: Sync + Send {
    fn as_vulkan(self: Arc<Self>) -> Option<Arc<vulkan::VulkanPipeline>> {
        None
    }
}

pub trait GpuVec {
    fn copy_from_slice(&mut self, data: &[u8]);
    fn read(&self) -> &[u8];
}

pub trait Buffer: Sync + Send {
    fn as_vulkan(self: Arc<Self>) -> Option<Arc<vulkan::BufferManager>> {
        None
    }

    fn get_memory(&self) -> Arc<Mutex<dyn GpuVec>>;
}
