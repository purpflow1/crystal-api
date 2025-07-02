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

#[derive(Clone)]
pub struct BufferInfo {
    pub size: u64,
    pub usage: vk::BufferUsageFlags,
    pub properties: vk::MemoryPropertyFlags,
}

pub struct BufferManager {
    device_manager: Arc<DeviceManager>,
    pub buffer: vk::Buffer,
    device_memory: vk::DeviceMemory,
    pub mapped_memory: RwLock<Option<*mut c_void>>,
    info: BufferInfo,
}

unsafe impl Sync for BufferManager {}
unsafe impl Send for BufferManager {}

impl Drop for BufferManager {
    fn drop(&mut self) {
        unsafe {
            self.device_manager.device.destroy_buffer(self.buffer, None);
            self.device_manager
                .device
                .free_memory(self.device_memory, None);
        }
    }
}

impl BufferManager {
    pub(crate) fn new(
        device_manager: Arc<DeviceManager>,
        info: BufferInfo,
    ) -> CrystalResult<Arc<Self>> {
        let create_info = vk::BufferCreateInfo::default()
            .size(info.size)
            .usage(info.usage)
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
            .find_memory_type_index(info.properties, memory_requirements.memory_type_bits)?;

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
            info,
        }))
    }

    pub fn map_memory(&self, size: u64, offset: u64) -> CrystalResult<*mut u8> {
        let mut mapped_memory = self.mapped_memory.write().unwrap();

        if mapped_memory.is_some() {
            unsafe { self.device_manager.device.unmap_memory(self.device_memory) };
        }

        *mapped_memory = match unsafe {
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

        Ok(mapped_memory.unwrap() as *mut u8)
    }

    pub fn unmap_memory(&self) -> CrystalResult<()> {
        let mut mapped = self.mapped_memory.write().unwrap();
        if mapped.is_some() {
            unsafe { self.device_manager.device.unmap_memory(self.device_memory) };
            *mapped = None;
            Ok(())
        } else {
            Err(CrystalError::MemoryError)
        }
    }

    pub fn write<T: Copy>(&self, data: &[T], offset: usize) -> CrystalResult<()> {
        let size = data.len() * size_of::<T>();

        assert!(
            size + offset <= self.info.size as usize,
            "fatal: copying outside of buffer size: {} > {}",
            size + offset,
            self.info.size
        );

        let lock = self.mapped_memory.read().unwrap();

        let mapped_memory = lock.unwrap_or(std::ptr::null_mut() as *mut c_void);

        assert!(
            !mapped_memory.is_null(),
            "fatal: trying to write to unmapped buffer"
        );

        unsafe {
            let ptr = mapped_memory.byte_add(offset * size_of::<T>());
            ptr.copy_from(
                data.as_ptr() as *const std::ffi::c_void,
                data.len() * size_of::<T>(),
            );
        }

        Ok(())
    }
}
