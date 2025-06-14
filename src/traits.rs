use std::{
    cell::{Ref, RefCell},
    sync::Arc,
};

use crate::{
    errors::CrystalResult,
    images::Image2D,
    mesh::Attribute,
    object::Object,
    shader::Shader,
    vulkan::{self, VulkanRenderTarget},
};

pub trait GraphicsApi {
    fn render(
        &mut self,
        layouts: &[Arc<RefCell<dyn Layout>>],
        render_target: Arc<RefCell<dyn RenderTarget>>,
    ) -> CrystalResult<()>;

    fn create_layout_from_data(
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
}

pub trait RenderTarget {
    fn create_graphics_pipeline(
        &self,
        layout: Ref<dyn Layout>,
        shaders: &[Shader],
        attributes: &[Attribute],
    ) -> CrystalResult<Arc<dyn Pipeline>>;

    fn update_size(&mut self, extent: [u32; 2]) -> CrystalResult<()>;

    fn as_vulkan_mut(&mut self) -> Option<&mut VulkanRenderTarget> {
        None
    }
}

pub trait Layout {
    fn as_vulkan_mut(&mut self) -> Option<&mut vulkan::VulkanLayout> {
        None
    }

    fn as_vulkan_ref(&self) -> Option<&vulkan::VulkanLayout> {
        None
    }

    fn add_object_to_queue(&mut self, object: Arc<RefCell<Object>>);
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
    fn as_vulkan(self: Arc<Self>) -> Option<Arc<vulkano::pipeline::GraphicsPipeline>> {
        None
    }
}
