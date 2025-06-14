use std::sync::Arc;

use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    device::DeviceOwned,
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
};

use crate::{
    debug::log,
    errors::{CrystalError, CrystalResult},
};

pub struct BufferManager<T: BufferContents> {
    pub buffer: Arc<Subbuffer<[T]>>,
}

impl<T: BufferContents + Clone> BufferManager<T> {
    pub(crate) fn new(
        memory_allocator: Arc<StandardMemoryAllocator>,
        data: Vec<T>,
        usage: BufferUsage,
        memory_type_filter: MemoryTypeFilter,
    ) -> CrystalResult<Self> {
        let mut create_info = BufferCreateInfo {
            usage,
            size: data.len() as u64 * size_of::<T>() as u64,
            ..Default::default()
        };

        let memory_requirements = match memory_allocator
            .device()
            .buffer_memory_requirements(create_info.clone())
        {
            Ok(req) => req,
            Err(e) => {
                log!("cannot get buffer memory requirements: {:?}", e);
                return Err(CrystalError::MemoryError);
            }
        };

        create_info.size = 0;

        let alloc_info = AllocationCreateInfo {
            memory_type_filter,
            memory_type_bits: memory_requirements.memory_type_bits,
            ..Default::default()
        };

        let buffer: Arc<Subbuffer<[T]>> =
            match Buffer::from_iter(memory_allocator.clone(), create_info, alloc_info, data) {
                Ok(buffer) => Arc::new(buffer),
                Err(e) => {
                    log!("cannot create vertex buffer: {:?}", e);
                    return Err(CrystalError::MemoryError);
                }
            };

        Ok(Self { buffer })
    }

    pub(crate) fn new_uninit(
        memory_allocator: Arc<StandardMemoryAllocator>,
        len: u64,
        usage: BufferUsage,
        memory_type_filter: MemoryTypeFilter,
    ) -> CrystalResult<Self> {
        let mut create_info = BufferCreateInfo {
            usage,
            size: len * size_of::<T>() as u64,
            ..Default::default()
        };

        let memory_requirements = match memory_allocator
            .device()
            .buffer_memory_requirements(create_info.clone())
        {
            Ok(req) => req,
            Err(e) => {
                log!("cannot get buffer memory requirements: {:?}", e);
                return Err(CrystalError::MemoryError);
            }
        };

        create_info.size = 0;

        let alloc_info = AllocationCreateInfo {
            memory_type_filter,
            memory_type_bits: memory_requirements.memory_type_bits,
            ..Default::default()
        };

        let buffer: Arc<Subbuffer<[T]>> =
            match Buffer::new_unsized(memory_allocator.clone(), create_info, alloc_info, len) {
                Ok(buffer) => Arc::new(buffer),
                Err(e) => {
                    log!("cannot create vertex buffer: {:?}", e);
                    return Err(CrystalError::MemoryError);
                }
            };

        Ok(Self { buffer })
    }

    pub fn write(&self, data: &[T], _offset: usize) -> CrystalResult<()> {
        // TODO offset doesn't work
        match self.buffer.write() {
            Ok(mut write_guard) => {
                write_guard.clone_from_slice(data);
            }
            Err(e) => {
                log!("cannot write to buffer: {:?}", e);
                return Err(CrystalError::MemoryError);
            }
        }

        Ok(())
    }
}
