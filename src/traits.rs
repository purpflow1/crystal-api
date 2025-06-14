use std::{
    cell::{Ref, RefCell},
    sync::Arc,
};

use crate::{
    GpuVec, errors::CrystalResult, images::Image2D, mesh::Attribute, object::Object,
    shader::Shader, vulkan,
};

pub trait GraphicsApi {
    fn render(&mut self, layouts: &[Arc<RefCell<dyn Layout>>]) -> CrystalResult<()>;

    fn create_layout(
        &self,
        frames_in_flight: u32,
        image_sampled_num: u32,
        max_instance_num: u64,
        buffers: &[(bool, u64)],
    ) -> CrystalResult<Arc<RefCell<dyn Layout>>>;

    fn create_texture(
        &self,
        image: &Image2D,
        anisotropy_texels: f32,
    ) -> CrystalResult<Arc<dyn Texture>>;

    fn get_viewport(&self) -> Arc<RefCell<dyn RenderTarget>>;
    fn get_current_frame(&self) -> usize;
}

pub trait RenderTarget {
    fn create_graphics_pipeline(
        &self,
        layout: Ref<dyn Layout>,
        shaders: &[Shader],
        attributes: &[Attribute],
    ) -> CrystalResult<Arc<dyn Pipeline>>;

    fn update_size(&mut self, width: u32, height: u32) -> CrystalResult<()>;
}

pub trait Layout {
    fn as_vulkan_mut(&mut self) -> Option<&mut vulkan::VulkanLayout> {
        None
    }

    fn as_vulkan_ref(&self) -> Option<&vulkan::VulkanLayout> {
        None
    }

    fn add_object_to_queue(&mut self, object: Arc<RefCell<Object>>);

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
    fn as_vulkan_arc(self: Arc<Self>) -> Option<Arc<vulkan::VulkanTexture>> {
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
    fn as_vulkan_mut(&mut self) -> Option<&mut ash::vk::Pipeline> {
        None
    }

    fn as_vulkan_ref(&self) -> Option<&ash::vk::Pipeline> {
        None
    }
}
