use std::{
    ffi::c_void,
    sync::{Arc, RwLock},
};

use ash::vk;

use crate::{
    debug::log,
    errors::{CrystalError, CrystalResult},
};

use super::devices::DeviceManager;

pub struct BufferManager {
    device_manager: Arc<DeviceManager>,
    pub buffer: vk::Buffer,
    device_memory: vk::DeviceMemory,
    mapped_memory: RwLock<Option<*mut c_void>>,
    size: u64,
}

unsafe impl Sync for BufferManager {}
unsafe impl Send for BufferManager {}

impl Drop for BufferManager {
    fn drop(&mut self) {
        unsafe {
            if let Some(_) = *self.mapped_memory.read().unwrap() {
                self.device_manager.device.unmap_memory(self.device_memory);
            }
            self.device_manager
                .device
                .free_memory(self.device_memory, None);
            self.device_manager.device.destroy_buffer(self.buffer, None);
        }
    }
}

impl BufferManager {
    pub(crate) fn new(
        device_manager: Arc<DeviceManager>,
        size: u64,
        usage: vk::BufferUsageFlags,
        properties: vk::MemoryPropertyFlags,
    ) -> CrystalResult<Arc<Self>> {
        let create_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = match unsafe { device_manager.device.create_buffer(&create_info, None) } {
            Ok(buffer) => buffer,
            Err(e) => {
                log!("cannot create vertex buffer: {}", e);
                return Err(CrystalError::MemoryError);
            }
        };

        let memory_requirements =
            unsafe { device_manager.device.get_buffer_memory_requirements(buffer) };

        let memory_type_index = device_manager
            .find_memory_type_index(properties, memory_requirements.memory_type_bits)?;

        let memory_allocate_info = vk::MemoryAllocateInfo::default()
            .allocation_size(memory_requirements.size)
            .memory_type_index(memory_type_index);

        let device_memory = match unsafe {
            device_manager
                .device
                .allocate_memory(&memory_allocate_info, None)
        } {
            Ok(device_memory) => device_memory,
            Err(e) => {
                log!("cannot allocate device memory: {}", e);
                return Err(CrystalError::MemoryError);
            }
        };

        match unsafe {
            device_manager
                .device
                .bind_buffer_memory(buffer, device_memory, 0)
        } {
            Ok(_) => (),
            Err(e) => {
                log!("cannot bind buffer memory: {}", e);
                return Err(CrystalError::MemoryError);
            }
        };

        Ok(Arc::new(Self {
            device_manager,
            buffer,
            device_memory,
            mapped_memory: RwLock::new(None),
            size,
        }))
    }

    pub fn map_memory(&self, size: u64, offset: u64) -> CrystalResult<()> {
        if self.mapped_memory.read().unwrap().is_some() {
            unsafe { self.device_manager.device.unmap_memory(self.device_memory) };
        }

        *self.mapped_memory.write().unwrap() = match unsafe {
            self.device_manager.device.map_memory(
                self.device_memory,
                offset,
                size,
                vk::MemoryMapFlags::empty(),
            )
        } {
            Ok(ptr) => Some(ptr),
            Err(e) => {
                log!("cannot map memory: {}", e);
                return Err(CrystalError::MemoryError);
            }
        };

        Ok(())
    }

    pub fn write<T: Copy>(&self, data: &[T], offset: usize) -> CrystalResult<()> {
        let size = data.len() * size_of::<T>();

        assert!(
            size + offset <= self.size as usize,
            "fatal: copying outside of buffer size: {} > {}",
            size + offset,
            self.size
        );

        unsafe {
            let ptr = self
                .mapped_memory
                .read()
                .unwrap()
                .unwrap()
                .byte_add(offset * size_of::<T>());
            ptr.copy_from(
                data.as_ptr() as *const std::ffi::c_void,
                data.len() * size_of::<T>(),
            );
        }

        Ok(())
    }

    pub fn single_time_write<T: Copy>(&self, data: &[T], offset: u64) -> CrystalResult<()> {
        let len_to_copy = (data.len() * size_of::<T>()) as u64;

        if len_to_copy + offset as u64 > self.size {
            panic!(
                "fatal: copying outside of buffer size: {} > {}",
                len_to_copy + offset,
                self.size
            );
        }

        let ptr = match unsafe {
            self.device_manager.device.map_memory(
                self.device_memory,
                offset,
                len_to_copy,
                vk::MemoryMapFlags::empty(),
            )
        } {
            Ok(ptr) => ptr,
            Err(e) => {
                log!("cannot map memory: {}", e);
                return Err(CrystalError::MemoryError);
            }
        };

        unsafe {
            ptr.copy_from(
                data.as_ptr() as *const std::ffi::c_void,
                data.len() * size_of::<T>(),
            );
        }

        unsafe { self.device_manager.device.unmap_memory(self.device_memory) };

        Ok(())
    }
}
