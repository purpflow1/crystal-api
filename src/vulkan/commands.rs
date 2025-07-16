use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
    u64,
};

use ash::{
    prelude::VkResult,
    vk::{self, Handle},
};

use crate::{
    debug::log,
    errors::{CrystalError, CrystalResult},
    vulkan::{devices::DeviceManager, presentation::Presentation},
};

use super::{devices::Queue, sync::GpuSync};

pub struct GpuFuture {
    sync: Arc<Mutex<GpuSync>>,
    command_buffers: Mutex<Vec<Arc<CommandBuffer>>>,
}

unsafe impl Send for GpuFuture {}
unsafe impl Sync for GpuFuture {}

impl GpuFuture {
    fn buffers(sync: Arc<Mutex<GpuSync>>, buffers: Vec<Arc<CommandBuffer>>) -> Box<Self> {
        Box::new(Self {
            sync,
            command_buffers: Mutex::new(buffers),
        })
    }

    /// Transfers other's buffers to self
    pub fn join(self: Box<Self>, other: Box<Self>) -> Box<Self> {
        let mut self_lock = self.command_buffers.lock().unwrap();
        let other_lock = other.command_buffers.lock().unwrap();
        other_lock
            .iter()
            .for_each(|oth| self_lock.push(oth.clone()));
        drop(self_lock);
        drop(other_lock);

        self
    }

    pub fn acquire_next_image(&self, presentation: &Presentation) -> VkResult<()> {
        let mut sync = self.sync.lock().unwrap();

        sync.flip();

        let (image_index, suboptimal) = unsafe {
            presentation
                .swapchain
                .swapchain
                .write()
                .unwrap()
                .acquire_next_image(
                    *presentation.swapchain.swapchain_khr.read().unwrap(),
                    u64::MAX,
                    sync.semaphore_image(),
                    vk::Fence::null(),
                )?
        };

        sync.image_index = image_index;

        if suboptimal {
            sync.unflip();
            return Err(vk::Result::SUBOPTIMAL_KHR);
        }

        Ok(())
    }

    pub fn flush(self: Box<Self>, queue: Arc<Queue>) -> CrystalResult<Box<Self>> {
        let mut command_buffer_lock = self.command_buffers.lock().unwrap();
        let sync_lock = self.sync.lock().unwrap();

        let command_buffers: Vec<vk::CommandBuffer> = (*command_buffer_lock)
            .iter()
            .map(|buffer| buffer.handler)
            .collect();

        let mut signal_semaphores = Vec::with_capacity(1);
        let mut fence = vk::Fence::null();

        if sync_lock.is_sync() {
            signal_semaphores.push(sync_lock.semaphore_transfer());
        } else {
            match unsafe {
                queue
                    .device
                    .create_fence(&vk::FenceCreateInfo::default(), None)
            } {
                Ok(f) => fence = f,
                Err(e) => {
                    log!("failed to create one time fence: {:?}", e);
                    return Err(CrystalError::SyncError);
                }
            };
        }

        let submit_info = vk::SubmitInfo::default()
            .command_buffers(&command_buffers)
            .signal_semaphores(&signal_semaphores);

        queue.submit(&[submit_info], fence).unwrap();

        if !fence.is_null() {
            unsafe {
                queue
                    .device
                    .wait_for_fences(&[fence], true, u64::MAX)
                    .unwrap();

                queue.device.destroy_fence(fence, None);
            };
        }

        drop(command_buffers);
        command_buffer_lock.clear();
        drop(command_buffer_lock);
        drop(sync_lock);

        Ok(self)
    }

    pub fn swapchain_present_and_flush(
        self: Box<Self>,
        queue: Arc<Queue>,
        presentation: Arc<Presentation>,
    ) -> Result<Box<Self>, (vk::Result, Arc<Mutex<GpuSync>>)> {
        let swapchain = presentation.swapchain.clone();

        let sync = self.sync.lock().unwrap();

        let wait_semaphores = [sync.semaphore_image(), sync.semaphore_transfer()];
        let render_semaphores = [sync.semaphore_render()];

        let swaphchains = [*swapchain.swapchain_khr.read().unwrap()];
        let indices = [sync.image_index];
        let mut command_buffers_lock = self.command_buffers.lock().unwrap();

        let command_buffers: Vec<vk::CommandBuffer> = command_buffers_lock
            .iter()
            .map(|cb| cb.handler)
            .filter(|h| !h.is_null())
            .collect();

        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(&wait_semaphores)
            .signal_semaphores(&render_semaphores)
            .wait_dst_stage_mask(&[
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::COMPUTE_SHADER,
            ])
            .command_buffers(&command_buffers);

        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&render_semaphores)
            .swapchains(&swaphchains)
            .image_indices(&indices);

        let result;

        //sync.wait_transfer().unwrap();

        let queue_lock = queue
            .submit_still_lock(&[submit_info], sync.fence_render())
            .unwrap();

        unsafe {
            command_buffers_lock.clear();
            drop(command_buffers_lock);

            result = swapchain
                .swapchain
                .write()
                .unwrap()
                .queue_present(*queue_lock, &present_info);
        };

        drop(queue_lock);

        if let Err(e) = result {
            return Err((e, self.sync.clone()));
        }

        drop(sync);

        Ok(self)
    }
}

pub struct CommandBuffer {
    pool: Arc<CommandPool>,
    handler: vk::CommandBuffer,
}

impl Drop for CommandBuffer {
    fn drop(&mut self) {
        unsafe {
            self.pool.queue.wait_idle().unwrap();
            self.pool
                .device_manager
                .device
                .free_command_buffers(self.pool.handler, &[self.handler]);
        }
    }
}

impl CommandBuffer {
    fn from_handlers(
        pool: Arc<CommandPool>,
        handlers: Vec<vk::CommandBuffer>,
    ) -> CrystalResult<Vec<Arc<Self>>> {
        Ok(handlers
            .iter()
            .map(|handler| {
                Arc::new(Self {
                    pool: pool.clone(),
                    handler: *handler,
                })
            })
            .collect())
    }

    fn new(
        pool: Arc<CommandPool>,
        buffer_count: u32,
        level: vk::CommandBufferLevel,
    ) -> CrystalResult<Vec<Arc<Self>>> {
        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool.handler)
            .level(level)
            .command_buffer_count(buffer_count);

        let command_buffers = match unsafe {
            pool.device_manager
                .device
                .allocate_command_buffers(&allocate_info)
        } {
            Ok(command_buffers) => command_buffers
                .iter()
                .map(|command_buffer| {
                    Arc::new(Self {
                        pool: pool.clone(),
                        handler: *command_buffer,
                    })
                })
                .collect(),
            Err(e) => {
                log!("cannot allocate command buffers: {}", e);
                return Err(CrystalError::CannotCreateCommandManager);
            }
        };

        Ok(command_buffers)
    }
}

struct CommandPool {
    device_manager: Arc<DeviceManager>,
    queue: Arc<Queue>,
    handler: vk::CommandPool,
}

impl Drop for CommandPool {
    fn drop(&mut self) {
        unsafe {
            self.device_manager
                .device
                .destroy_command_pool(self.handler, None);
        }
    }
}

impl CommandPool {
    fn new(device_manager: Arc<DeviceManager>, queue: Arc<Queue>) -> CrystalResult<Arc<Self>> {
        let create_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(queue.family_index);

        let command_pool = match unsafe {
            device_manager
                .device
                .create_command_pool(&create_info, None)
        } {
            Ok(command_pool) => command_pool,
            Err(e) => {
                log!("cannot create command pool: {}", e);
                return Err(CrystalError::CannotCreateCommandManager);
            }
        };

        Ok(Arc::new(Self {
            queue,
            device_manager: device_manager.clone(),
            handler: command_pool,
        }))
    }
}

pub struct CommandEntry {
    device_manager: Arc<DeviceManager>,
    command_pool: Arc<CommandPool>,
    command_buffers: Vec<Arc<CommandBuffer>>,
    pub queue: Arc<Queue>,
}

impl CommandEntry {
    pub fn now(&self, sync: Arc<Mutex<GpuSync>>) -> Box<GpuFuture> {
        GpuFuture::buffers(sync, vec![])
    }

    fn new(
        device_manager: Arc<DeviceManager>,
        queue: Arc<Queue>,
        buffer_count: u32,
    ) -> CrystalResult<Self> {
        let command_pool = CommandPool::new(device_manager.clone(), queue.clone())?;

        let command_buffers = CommandBuffer::new(
            command_pool.clone(),
            buffer_count,
            vk::CommandBufferLevel::PRIMARY,
        )?;

        Ok(Self {
            device_manager,
            command_pool,
            command_buffers,
            queue,
        })
    }

    pub fn wait(&self) -> CrystalResult<()> {
        self.queue.wait_idle()
    }

    pub fn record_single_time_buffer<P>(&self, predicate: P) -> CrystalResult<Box<GpuFuture>>
    where
        P: Fn(&vk::CommandBuffer, Arc<ash::Device>),
    {
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_pool(self.command_pool.handler)
            .command_buffer_count(1);

        let command_buffer = match unsafe {
            self.device_manager
                .device
                .allocate_command_buffers(&alloc_info)
        } {
            Ok(buffers) => buffers[0],
            Err(e) => {
                log!("cannot allocate command buffer: {}", e);
                return Err(CrystalError::CommandManagerError);
            }
        };

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        match unsafe {
            self.device_manager
                .device
                .begin_command_buffer(command_buffer, &begin_info)
        } {
            Ok(_) => (),
            Err(e) => {
                log!("cannot begin command buffer: {}", e);
                return Err(CrystalError::CommandManagerError);
            }
        }

        predicate(&command_buffer, self.device_manager.device.clone());

        match unsafe {
            self.device_manager
                .device
                .end_command_buffer(command_buffer)
        } {
            Ok(()) => (),
            Err(e) => {
                log!("cannot begin command buffer: {}", e);
                return Err(CrystalError::CommandManagerError);
            }
        }

        let commands_buffers = vec![command_buffer];

        let command_buffer_managers =
            CommandBuffer::from_handlers(self.command_pool.clone(), commands_buffers)?;

        Ok(GpuFuture::buffers(
            GpuSync::no_sync(self.device_manager.clone()),
            command_buffer_managers,
        ))
    }

    pub fn record_command_buffer<P>(
        &self,
        sync: Arc<Mutex<GpuSync>>,
        predicate: P,
    ) -> CrystalResult<Box<GpuFuture>>
    where
        P: Fn(&vk::CommandBuffer, Arc<ash::Device>, usize),
    {
        let lock = sync.lock().unwrap();
        let n_pass = lock.image_index as usize;

        match unsafe {
            self.device_manager.device.reset_command_buffer(
                self.command_buffers[n_pass].handler,
                vk::CommandBufferResetFlags::RELEASE_RESOURCES,
            )
        } {
            Ok(()) => {}
            Err(e) => {
                log!("failed resetting command buffer: {}", e);
                return Err(CrystalError::CommandManagerError);
            }
        };

        let begin_info = vk::CommandBufferBeginInfo::default();

        match unsafe {
            self.device_manager
                .device
                .begin_command_buffer(self.command_buffers[n_pass].handler, &begin_info)
        } {
            Ok(_) => {}
            Err(e) => {
                log!("cannot begin command buffer: {}", e);
                return Err(CrystalError::CommandManagerError);
            }
        };

        predicate(
            &self.command_buffers[n_pass].handler,
            self.device_manager.device.clone(),
            n_pass,
        );

        match unsafe {
            self.device_manager
                .device
                .end_command_buffer(self.command_buffers[n_pass].handler)
        } {
            Ok(_) => {}
            Err(e) => {
                log!("cannot end command buffer: {}", e);
                return Err(CrystalError::CommandManagerError);
            }
        };

        drop(lock);

        Ok(GpuFuture::buffers(
            sync.clone(),
            vec![self.command_buffers[n_pass].clone()],
        ))
    }
}

#[derive(PartialEq, Eq, PartialOrd, Ord)]
pub enum CommandType {
    Graphics,
    Transfer,
    Compute,
}

pub struct CommandManager {
    pub device_manager: Arc<DeviceManager>,

    pub command_entries: BTreeMap<CommandType, Arc<CommandEntry>>,
}

impl CommandManager {
    pub(crate) fn new(
        device_manager: Arc<DeviceManager>,
        buffer_count: u32,
    ) -> CrystalResult<Arc<Self>> {
        let mut command_entries = BTreeMap::<CommandType, Arc<CommandEntry>>::new();

        if let Some(queue) = device_manager
            .queues
            .iter()
            .find(|queue| queue.flags.intersects(vk::QueueFlags::GRAPHICS))
        {
            command_entries.insert(
                CommandType::Graphics,
                Arc::new(CommandEntry::new(
                    device_manager.clone(),
                    queue.clone(),
                    buffer_count,
                )?),
            );
        }

        if let Some(queue) = device_manager
            .queues
            .iter()
            .find(|queue| queue.flags.intersects(vk::QueueFlags::TRANSFER))
        {
            command_entries.insert(
                CommandType::Transfer,
                Arc::new(CommandEntry::new(
                    device_manager.clone(),
                    queue.clone(),
                    buffer_count,
                )?),
            );
        }

        if let Some(queue) = device_manager
            .queues
            .iter()
            .find(|queue| queue.flags.intersects(vk::QueueFlags::COMPUTE))
        {
            command_entries.insert(
                CommandType::Compute,
                Arc::new(CommandEntry::new(
                    device_manager.clone(),
                    queue.clone(),
                    buffer_count,
                )?),
            );
        }

        Ok(Arc::new(Self {
            device_manager,
            command_entries,
        }))
    }
}
