use std::sync::Arc;

use ash::vk;

use crate::errors::CrystalResult;
use crate::mesh::{Index, VertexTexture};
use crate::traits;

use super::devices::DeviceManager;
use super::memory::BufferManager;

pub struct VulkanObjectMemoryManager {
    pub vertex_buffer_manager: BufferManager,
    pub index_buffer_manager: BufferManager,
}

impl traits::ObjectMemoryManager for VulkanObjectMemoryManager {
    fn as_vulkan_mut(&mut self) -> Option<&mut super::VulkanObjectMemoryManager> {
        Some(self)
    }

    fn as_vulkan_ref(&self) -> Option<&super::VulkanObjectMemoryManager> {
        Some(self)
    }
}

impl VulkanObjectMemoryManager {
    pub fn new(
        device_manager: Arc<DeviceManager>,
        vertices: &[VertexTexture],
        indices: &[Index],
    ) -> CrystalResult<Box<dyn traits::ObjectMemoryManager>> {
        let vertex_buffer_manager = BufferManager::new(
            device_manager.clone(),
            (vertices.len() * size_of::<VertexTexture>()) as u64,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        vertex_buffer_manager.single_time_write(vertices, 0)?;

        let index_buffer_manager = BufferManager::new(
            device_manager.clone(),
            (indices.len() * size_of::<Index>()) as u64,
            vk::BufferUsageFlags::INDEX_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        index_buffer_manager.single_time_write(indices, 0)?;

        Ok(Box::new(Self {
            vertex_buffer_manager,
            index_buffer_manager,
        }))
    }
}
