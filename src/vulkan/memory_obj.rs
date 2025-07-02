use std::sync::Arc;

use ash::vk;

use crate::errors::CrystalResult;
use crate::mesh::{Index, VertexTexture};
use crate::traits;

use super::devices::DeviceManager;
use super::memory::{BufferInfo, BufferManager};

pub struct VulkanObjectMemoryManager {
    pub vertex_buffer_manager: Arc<BufferManager>,
    pub index_buffer_manager: Arc<BufferManager>,
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
        let vertex_size = (vertices.len() * size_of::<VertexTexture>()) as u64;

        let mut buffer_info = BufferInfo {
            size: vertex_size,
            usage: vk::BufferUsageFlags::VERTEX_BUFFER,
            properties: vk::MemoryPropertyFlags::HOST_VISIBLE
                | vk::MemoryPropertyFlags::HOST_COHERENT,
        };

        let vertex_buffer_manager =
            BufferManager::new(device_manager.clone(), buffer_info.clone())?;

        vertex_buffer_manager.map_memory(vertex_size, 0).unwrap();
        vertex_buffer_manager.write(vertices, 0)?;
        vertex_buffer_manager.unmap_memory().unwrap();

        let index_size = (indices.len() * size_of::<Index>()) as u64;

        buffer_info.usage = vk::BufferUsageFlags::INDEX_BUFFER;
        buffer_info.size = index_size;

        let index_buffer_manager = BufferManager::new(device_manager.clone(), buffer_info)?;

        index_buffer_manager.map_memory(index_size, 0).unwrap();
        index_buffer_manager.write(indices, 0)?;
        index_buffer_manager.unmap_memory().unwrap();

        Ok(Box::new(Self {
            vertex_buffer_manager,
            index_buffer_manager,
        }))
    }
}
