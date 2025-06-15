use std::{cell::RefCell, sync::Arc};

use crate::{
    GpuVec, errors::CrystalResult, images::Image2D, mesh::Attribute, object::Object,
    shader::Shader, vulkan,
};

pub trait GraphicsApi {
    fn render_and_present(&mut self, layouts: Vec<Arc<dyn Layout>>) -> CrystalResult<()>;

    fn create_layout(
        &self,
        frames_in_flight: u32,
        image_sampled_num: u32,
        max_instance_num: u64,
        buffers: &[(bool, u64)],
    ) -> CrystalResult<Arc<dyn Layout>>;

    fn create_texture(
        &self,
        image: &Image2D,
        anisotropy_texels: f32,
    ) -> CrystalResult<Arc<dyn Texture>>;

    fn get_viewport(&self) -> Arc<dyn RenderTarget>;
    fn get_current_frame(&self) -> usize;
}

pub trait RenderTarget {
    fn create_graphics_pipeline(
        &self,
        layout: Arc<dyn Layout>,
        shaders: &[Shader],
        attributes: &[Attribute],
    ) -> CrystalResult<Arc<dyn Pipeline>>;

    fn update_size(&self, width: u32, height: u32) -> CrystalResult<()>;

    fn as_vulkan(self: Arc<Self>) -> Option<Arc<vulkan::VulkanRenderTarget>> {
        None
    }
}

pub trait Layout {
    fn as_vulkan(self: Arc<Self>) -> Option<Arc<vulkan::VulkanLayout>> {
        None
    }

    fn add_object_to_queue(&self, object: Arc<RefCell<Object>>);

    fn write_to_buffer(
        &self,
        is_uniform: bool,
        frame: usize,
        buffer: usize,
        offset: usize,
        data: GpuVec,
    ) -> CrystalResult<()>;
}

pub trait Texture {
    fn as_vulkan(self: Arc<Self>) -> Option<Arc<vulkan::VulkanTexture>> {
        None
    }
}

pub(crate) trait ObjectMemoryManager {
    fn as_vulkan_mut(&mut self) -> Option<&mut vulkan::VulkanObjectMemoryManager> {
        None
    }

    fn as_vulkan_ref(&self) -> Option<&vulkan::VulkanObjectMemoryManager> {
        None
    }
}

pub trait Pipeline {
    fn as_vulkan(self: Arc<Self>) -> Option<Arc<vulkan::VulkanPipeline>> {
        None
    }
}
