use std::sync::Arc;

use ash::vk;

use crate::debug::log;
use crate::errors::CrystalResult;
use crate::gpu_data::AsBytes;
use crate::mesh::{Index, VertexTexture};
use crate::{Buffer, traits};

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
        let index_size = (indices.len() * size_of::<Index>()) as u64;
        log!(
            "creating object data of {:.1} MB",
            (vertex_size + index_size) as f32 / 1024. / 1024.
        );

        let mut buffer_info = BufferInfo {
            size: vertex_size,
            usage: vk::BufferUsageFlags::VERTEX_BUFFER,
            properties: vk::MemoryPropertyFlags::HOST_VISIBLE
                | vk::MemoryPropertyFlags::HOST_COHERENT,
            count: 1,
        };

        let vertex_buffer_manager =
            BufferManager::new(device_manager.clone(), buffer_info.clone(), None)?;

        vertex_buffer_manager
            .get_memory_full()
            .copy_from_slice(vertices.as_bytes());

        buffer_info.usage = vk::BufferUsageFlags::INDEX_BUFFER;
        buffer_info.size = index_size;

        let index_buffer_manager = BufferManager::new(device_manager.clone(), buffer_info, None)?;

        index_buffer_manager
            .get_memory_full()
            .copy_from_slice(indices.as_bytes());

        Ok(Box::new(Self {
            vertex_buffer_manager,
            index_buffer_manager,
        }))
    }
}
