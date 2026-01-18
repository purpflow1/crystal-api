use std::{
    ops::Range,
    sync::{Arc, Mutex},
};

use ash::vk;

use crate::{
    debug::error,
    errors::{GraphicsError, GraphicsResult},
    proxies,
};

use super::{devices::DeviceManager, sync::GpuSync};

#[derive(Clone)]
pub(crate) struct BufferInfo {
    pub size: u64,
    pub usage: vk::BufferUsageFlags,
    pub properties: vk::MemoryPropertyFlags,
    pub count: usize,
}

pub(crate) struct BufferData {
    handler: vk::Buffer,
    memory: vk::DeviceMemory,
    mapped: *mut u8,
}

impl BufferData {
    fn new(device_manager: Arc<DeviceManager>, info: BufferInfo) -> GraphicsResult<Self> {
        let create_info = vk::BufferCreateInfo::default()
            .size(info.size)
            .usage(info.usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = match unsafe { device_manager.device.create_buffer(&create_info, None) } {
            Ok(buffer) => buffer,
            Err(e) => {
                error!("cannot create vertex buffer: {}", e);
                return Err(GraphicsError::MemoryError);
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
                error!("cannot allocate device memory: {}", e);
                return Err(GraphicsError::MemoryError);
            }
        };

        match unsafe {
            device_manager
                .device
                .bind_buffer_memory(buffer, device_memory, 0)
        } {
            Ok(_) => (),
            Err(e) => {
                error!("cannot bind buffer memory: {}", e);
                return Err(GraphicsError::MemoryError);
            }
        };

        let mapped = match unsafe {
            device_manager.device.map_memory(
                device_memory,
                0,
                info.size,
                vk::MemoryMapFlags::empty(),
            )
        } {
            Ok(ptr) => ptr as *mut u8,
            Err(e) => {
                error!("cannot map memory: {}", e);
                return Err(GraphicsError::MemoryError);
            }
        };

        Ok(Self {
            handler: buffer,
            memory: device_memory,
            mapped,
        })
    }
}

pub(crate) struct BufferManager {
    device_manager: Arc<DeviceManager>,
    buffer_data: Vec<BufferData>,
    pub info: BufferInfo,
    sync: Arc<Mutex<GpuSync>>,
}

unsafe impl Sync for BufferManager {}
unsafe impl Send for BufferManager {}

impl Drop for BufferManager {
    fn drop(&mut self) {
        // TODO temporary added wait_idle due to
        // buffer destroying which are used by command buffers
        self.sync
            .lock()
            .unwrap()
            .device_manager
            .wait_idle()
            .unwrap();

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

impl proxies::BufferProxy for BufferManager {
    #[inline]
    fn as_vulkan(self: Arc<Self>) -> Option<Arc<super::BufferManager>> {
        Some(self.clone())
    }

    #[inline]
    fn get_size(&self) -> u64 {
        self.info.size
    }

    #[inline]
    fn get_memory<'a>(&self, range: Range<u64>) -> &'a mut [u8] {
        let lock = self.sync.lock().unwrap();
        let idx = if self.info.count > 1 {
            if lock.is_odd_frame { 0 } else { 1 }
        } else {
            0
        };

        unsafe {
            let ptr = self.buffer_data[idx].mapped.byte_add(range.start as usize);
            std::slice::from_raw_parts_mut(ptr, (range.end - range.start) as usize)
        }
    }
}

impl BufferManager {
    pub(crate) fn new(
        device_manager: Arc<DeviceManager>,
        info: BufferInfo,
        sync: Option<Arc<Mutex<GpuSync>>>,
    ) -> GraphicsResult<Arc<Self>> {
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

    pub(crate) fn get_handlers(&self) -> Vec<vk::Buffer> {
        self.buffer_data.iter().map(|buf| buf.handler).collect()
    }
}
