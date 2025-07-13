use std::{
    mem::ManuallyDrop,
    sync::{Arc, Mutex},
};

use ash::vk;

use crate::{
    debug::log,
    errors::{CrystalError, CrystalResult},
    traits,
};

use super::{devices::DeviceManager, sync::GpuSync};

#[repr(transparent)]
pub struct VulkanGpuVec(ManuallyDrop<Vec<u8>>);

impl traits::GpuVec for VulkanGpuVec {
    fn copy_from_slice(&mut self, data: &[u8]) {
        (self.0[0..data.len()]).copy_from_slice(data);
    }

    fn read(&self) -> &[u8] {
        &self.0
    }
}

impl VulkanGpuVec {
    pub(crate) fn from_raw(ptr: *mut u8, size: usize) -> Self {
        let tmp = unsafe { Vec::from_raw_parts(ptr, size, size) };
        Self(ManuallyDrop::new(tmp))
    }

    pub(crate) fn is_null(&self) -> bool {
        self.0.capacity() == 0
    }

    pub(crate) fn empty() -> Self {
        let tmp = Vec::with_capacity(0);
        Self(ManuallyDrop::new(tmp))
    }
}

#[derive(Clone)]
pub struct BufferInfo {
    pub size: u64,
    pub usage: vk::BufferUsageFlags,
    pub properties: vk::MemoryPropertyFlags,
    pub count: usize,
}

pub struct BufferData {
    device_manager: Arc<DeviceManager>,
    handler: vk::Buffer,
    memory: vk::DeviceMemory,
    mapped: Arc<Mutex<VulkanGpuVec>>,
}

impl BufferData {
    fn new(device_manager: Arc<DeviceManager>, info: BufferInfo) -> CrystalResult<Self> {
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

        Ok(Self {
            device_manager,
            handler: buffer,
            memory: device_memory,
            mapped: Arc::new(Mutex::new(VulkanGpuVec::empty())),
        })
    }

    fn map_memory(&self, size: u64, offset: u64) -> CrystalResult<Arc<Mutex<dyn traits::GpuVec>>> {
        let mut mapped = self.mapped.lock().unwrap();

        if !mapped.is_null() {
            drop(mapped);
            return Ok(self.mapped.clone());
        }

        let ptr = match unsafe {
            self.device_manager.device.map_memory(
                self.memory,
                offset,
                size,
                vk::MemoryMapFlags::empty(),
            )
        } {
            Ok(ptr) => ptr as *mut u8,
            Err(e) => {
                log!("cannot map memory: {}", e);
                return Err(CrystalError::MemoryError);
            }
        };

        *mapped = VulkanGpuVec::from_raw(ptr, size as usize);
        drop(mapped);

        Ok(self.mapped.clone())
    }
}

pub struct BufferManager {
    device_manager: Arc<DeviceManager>,
    buffer_data: Vec<BufferData>,
    pub info: BufferInfo,
    sync: Arc<Mutex<GpuSync>>,
}

unsafe impl Sync for BufferManager {}
unsafe impl Send for BufferManager {}

impl Drop for BufferManager {
    fn drop(&mut self) {
        for buffer_data in &self.buffer_data {
            unsafe {
                self.device_manager
                    .device
                    .destroy_buffer(buffer_data.handler, None);
                self.device_manager
                    .device
                    .free_memory(buffer_data.memory, None);
            }
        }
    }
}

impl traits::Buffer for BufferManager {
    fn as_vulkan(self: Arc<Self>) -> Option<Arc<super::BufferManager>> {
        Some(self.clone())
    }

    fn get_memory(&self) -> Arc<Mutex<dyn traits::GpuVec>> {
        let lock = self.sync.lock().unwrap();
        let idx = if self.info.count > 1 {
            lock.odd_pass
        } else {
            0
        };
        self.buffer_data[idx]
            .map_memory(self.info.size, 0)
            .expect("fatal: vulkan memory mapping error")
    }
}

impl BufferManager {
    pub(crate) fn new(
        device_manager: Arc<DeviceManager>,
        info: BufferInfo,
        sync: Option<Arc<Mutex<GpuSync>>>,
    ) -> CrystalResult<Arc<Self>> {
        let buffer_data = (0..info.count)
            .map(|_| BufferData::new(device_manager.clone(), info.clone()).unwrap())
            .collect();
        Ok(Arc::new(Self {
            device_manager: device_manager.clone(),
            buffer_data,
            info,
            sync: sync.unwrap_or(GpuSync::no_sync(device_manager)),
        }))
    }

    pub fn get_handlers(&self) -> Vec<vk::Buffer> {
        self.buffer_data.iter().map(|buf| buf.handler).collect()
    }
}
