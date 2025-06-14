use std::sync::Arc;

use vulkano::buffer::BufferUsage;
use vulkano::device::Device;
use vulkano::memory::allocator::{MemoryTypeFilter, StandardMemoryAllocator};

use super::memory::BufferManager;
use crate::errors::CrystalResult;
use crate::mesh::{Index, VertexTexture};
use crate::traits;

pub struct VulkanObjectMemoryManager {
    pub vertex_buffer_manager: BufferManager<VertexTexture>,
    pub index_buffer_manager: BufferManager<Index>,
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
        device: Arc<Device>,
        vertices: &[VertexTexture],
        indices: &[Index],
    ) -> CrystalResult<Box<dyn traits::ObjectMemoryManager>> {
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device));

        let vertex_buffer_manager = BufferManager::new(
            memory_allocator.clone(),
            vertices.to_vec(),
            BufferUsage::VERTEX_BUFFER,
            MemoryTypeFilter::HOST_SEQUENTIAL_WRITE | MemoryTypeFilter::PREFER_DEVICE,
        )?;

        let index_buffer_manager = BufferManager::new(
            memory_allocator.clone(),
            indices.to_vec(),
            BufferUsage::INDEX_BUFFER,
            MemoryTypeFilter::HOST_SEQUENTIAL_WRITE | MemoryTypeFilter::PREFER_DEVICE,
        )?;

        Ok(Box::new(Self {
            vertex_buffer_manager,
            index_buffer_manager,
        }))
    }
}
