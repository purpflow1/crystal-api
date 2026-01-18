use std::sync::{Arc, Mutex};

use ash::vk::{self};

use crate::{
    debug::error,
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

pub(crate) struct GpuSync {
    pub device_manager: Arc<DeviceManager>,
    barriers: Barriers,
    current_barrier_index: usize,
    pub is_odd_frame: bool,
    pub image_index: u32,
}

impl Drop for GpuSync {
    fn drop(&mut self) {
        self.device_manager.wait_idle().ok();

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
    pub(crate) fn no_sync(device_manager: Arc<DeviceManager>) -> Arc<Mutex<Self>> {
        Arc::new(Mutex::new(Self {
            device_manager,
            barriers: Default::default(),
            current_barrier_index: 0,
            is_odd_frame: false,
            image_index: 0,
        }))
    }

    pub(crate) fn new(
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
                        error!("cannot create semaphore: {}", e);
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
                        error!("cannot create semaphore: {}", e);
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
                        error!("cannot create semaphore: {}", e);
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
                error!("cannot create fence: {}", e);
                return Err(GraphicsError::SyncError);
            }
        };

        fence_render[1] =
            match unsafe { device_manager.device.create_fence(&fence_create_info, None) } {
                Ok(semaphore) => semaphore,
                Err(e) => {
                    error!("cannot create fence: {}", e);
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
            current_barrier_index: n_pass,
            is_odd_frame: true,
            image_index: 0,
        })))
    }

    pub(crate) fn flip(&mut self) {
        self.current_barrier_index =
            (self.current_barrier_index + 1) % self.barriers.semaphore_image.len();
        self.is_odd_frame = !self.is_odd_frame;
    }

    fn wait_fences(&self, fences: &[vk::Fence]) -> GraphicsResult<()> {
        unsafe {
            match self
                .device_manager
                .device
                .wait_for_fences(fences, true, u64::MAX)
            {
                Ok(()) => (),
                Err(e) => {
                    error!("cannot wait for fences: {e:?}");
                    return Err(GraphicsError::SyncError);
                }
            };

            match self.device_manager.device.reset_fences(fences) {
                Ok(()) => (),
                Err(e) => {
                    error!("cannot reset fences: {e:?}");
                    return Err(GraphicsError::SyncError);
                }
            }
        }
        Ok(())
    }

    pub(crate) fn wait_render(&self) -> GraphicsResult<()> {
        // TODO something wrong with vsync disabled
        self.wait_fences(&[self.barriers.fence_render[if self.is_odd_frame { 0 } else { 1 }]])
    }

    pub(crate) fn fence_render(&self) -> vk::Fence {
        self.barriers.fence_render[if self.is_odd_frame { 0 } else { 1 }]
    }

    pub(crate) fn semaphore_render(&self) -> vk::Semaphore {
        self.barriers.semaphore_render[self.image_index as usize]
    }

    pub(crate) fn semaphore_image(&self) -> vk::Semaphore {
        self.barriers.semaphore_image[self.current_barrier_index]
    }

    pub(crate) fn semaphore_transfer(&self) -> vk::Semaphore {
        self.barriers.semaphore_transfer[self.current_barrier_index]
    }

    pub(crate) fn is_sync(&self) -> bool {
        !self.barriers.semaphore_transfer.is_empty()
    }
}
