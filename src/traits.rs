use std::sync::{Arc, RwLock};

use crate::{
    GpuVec, errors::CrystalResult, mesh::Attribute, object::Object, shader::Shader, vulkan,
};

pub trait RenderTarget: Sync + Send {
    fn create_graphics_pipeline(
        &self,
        layout: Arc<dyn Layout>,
        shaders: &[Shader],
        attributes: &[Attribute],
    ) -> CrystalResult<Arc<dyn Pipeline>>;

    fn as_vulkan(self: Arc<Self>) -> Option<Arc<vulkan::VulkanRenderTarget>> {
        None
    }
}

pub trait Layout: Sync + Send {
    fn as_vulkan(self: Arc<Self>) -> Option<Arc<vulkan::VulkanLayout>> {
        None
    }

    fn add_object_to_queue(&self, object: Arc<RwLock<Object>>);

    fn write_to_buffer(
        &self,
        is_uniform: bool,
        frame: usize,
        buffer: usize,
        offset: usize,
        data: GpuVec,
    ) -> CrystalResult<()>;
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
