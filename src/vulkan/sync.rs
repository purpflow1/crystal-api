use std::{
    sync::{Arc, Mutex},
    u64,
};

use ash::vk::{self};

use crate::{
    debug::log,
    errors::{GraphicsError, GraphicsResult},
};

use super::devices::DeviceManager;

#[derive(Default)]
pub(crate) struct Barriers {
    pub semaphore_image: Vec<vk::Semaphore>,
    pub semaphore_render: Vec<vk::Semaphore>,
    pub semaphore_transfer: Vec<vk::Semaphore>,
    pub fence_render: [vk::Fence; 2],
}

pub struct GpuSync {
    pub device_manager: Arc<DeviceManager>,
    barriers: Barriers,
    n_pass: usize,
    pub odd_pass: usize,
    pub image_index: u32,
}

impl Drop for GpuSync {
    fn drop(&mut self) {
        self.device_manager.wait_idle().unwrap();

        unsafe {
            for i in 0..self.barriers.semaphore_image.len() {
                self.device_manager
                    .device
                    .destroy_semaphore(self.barriers.semaphore_image[i], None);
                self.device_manager
                    .device
                    .destroy_semaphore(self.barriers.semaphore_render[i], None);
                self.device_manager
                    .device
                    .destroy_semaphore(self.barriers.semaphore_transfer[i], None);
            }

            for i in 0..2 {
                self.device_manager
                    .device
                    .destroy_fence(self.barriers.fence_render[i], None);
            }
        }
    }
}

impl GpuSync {
    pub fn no_sync(device_manager: Arc<DeviceManager>) -> Arc<Mutex<Self>> {
        Arc::new(Mutex::new(Self {
            device_manager,
            barriers: Default::default(),
            n_pass: 0,
            odd_pass: 0,
            image_index: 0,
        }))
    }

    pub fn new(
        device_manager: Arc<DeviceManager>,
        render_images: u32,
    ) -> GraphicsResult<Arc<Mutex<Self>>> {
        let (semaphore_create_info, fence_create_info) = Default::default();

        let mut semaphore_image = Vec::with_capacity(render_images as usize);
        let mut semaphore_render = Vec::with_capacity(render_images as usize);
        let mut semaphore_transfer = Vec::with_capacity(render_images as usize);
        let mut fence_render = [vk::Fence::null(); 2];

        for _ in 0..render_images {
            semaphore_image.push(
                match unsafe {
                    device_manager
                        .device
                        .create_semaphore(&semaphore_create_info, None)
                } {
                    Ok(semaphore) => semaphore,
                    Err(e) => {
                        log!("cannot create semaphore: {}", e);
                        return Err(GraphicsError::SyncError);
                    }
                },
            );

            semaphore_render.push(
                match unsafe {
                    device_manager
                        .device
                        .create_semaphore(&semaphore_create_info, None)
                } {
                    Ok(semaphore) => semaphore,
                    Err(e) => {
                        log!("cannot create semaphore: {}", e);
                        return Err(GraphicsError::SyncError);
                    }
                },
            );

            semaphore_transfer.push(
                match unsafe {
                    device_manager
                        .device
                        .create_semaphore(&semaphore_create_info, None)
                } {
                    Ok(semaphore) => semaphore,
                    Err(e) => {
                        log!("cannot create semaphore: {}", e);
                        return Err(GraphicsError::SyncError);
                    }
                },
            );
        }

        fence_render[0] = match unsafe {
            device_manager.device.create_fence(
                &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                None,
            )
        } {
            Ok(semaphore) => semaphore,
            Err(e) => {
                log!("cannot create fence: {}", e);
                return Err(GraphicsError::SyncError);
            }
        };

        fence_render[1] =
            match unsafe { device_manager.device.create_fence(&fence_create_info, None) } {
                Ok(semaphore) => semaphore,
                Err(e) => {
                    log!("cannot create fence: {}", e);
                    return Err(GraphicsError::SyncError);
                }
            };

        let n_pass = semaphore_image.len() - 1;

        Ok(Arc::new(Mutex::new(Self {
            device_manager,
            barriers: Barriers {
                semaphore_image,
                semaphore_render,
                semaphore_transfer,
                fence_render,
            },
            n_pass,
            odd_pass: 0,
            image_index: 0,
        })))
    }

    pub fn unflip(&mut self) {
        self.n_pass = self.barriers.semaphore_image.len() - 1;
        self.odd_pass = 0;
    }

    pub fn flip(&mut self) {
        self.n_pass = (self.n_pass + 1) % self.barriers.semaphore_image.len();
        self.odd_pass = (self.odd_pass + 1) % 2;
    }

    fn wait_fences(&self, fences: &[vk::Fence]) -> GraphicsResult<()> {
        unsafe {
            self.device_manager
                .device
                .wait_for_fences(fences, true, u64::MAX)
                .unwrap();

            self.device_manager.device.reset_fences(fences).unwrap();
        }
        Ok(())
    }

    pub fn wait_render(&self) -> GraphicsResult<()> {
        self.wait_fences(&[self.barriers.fence_render[self.odd_pass]])
    }

    pub fn fence_render(&self) -> vk::Fence {
        self.barriers.fence_render[self.odd_pass]
    }

    pub fn semaphore_render(&self) -> vk::Semaphore {
        self.barriers.semaphore_render[self.image_index as usize]
    }

    pub fn semaphore_image(&self) -> vk::Semaphore {
        self.barriers.semaphore_image[self.n_pass]
    }

    pub fn semaphore_transfer(&self) -> vk::Semaphore {
        self.barriers.semaphore_transfer[self.n_pass as usize]
    }

    pub fn is_sync(&self) -> bool {
        self.barriers.semaphore_transfer.len() > 0
    }
}
