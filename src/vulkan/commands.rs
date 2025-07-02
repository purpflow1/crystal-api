use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
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

use super::devices::Queue;

pub struct GpuSync {
    device_manager: Arc<DeviceManager>,

    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,

    pub n_pass: usize,
}

impl GpuSync {
    fn new(device_manager: Arc<DeviceManager>) -> CrystalResult<Self> {
        let mut image_available_semaphores = vec![];
        let mut render_finished_semaphores = vec![];
        let mut in_flight_fences = vec![];

        for _ in 0..2 {
            let (semaphore_create_info, fence_create_info) = Default::default();

            for i in 0..2 {
                let semaphore = match unsafe {
                    device_manager
                        .device
                        .create_semaphore(&semaphore_create_info, None)
                } {
                    Ok(semaphore) => semaphore,
                    Err(e) => {
                        log!("cannot create semaphore: {}", e);
                        return Err(CrystalError::SyncError);
                    }
                };

                if i == 0 {
                    render_finished_semaphores.push(semaphore);
                } else {
                    image_available_semaphores.push(semaphore);
                }
            }

            let in_flight_fence =
                match unsafe { device_manager.device.create_fence(&fence_create_info, None) } {
                    Ok(fence) => fence,
                    Err(e) => {
                        log!("cannot create fence: {}", e);
                        return Err(CrystalError::SwapChainIsNotSupported);
                    }
                };

            in_flight_fences.push(in_flight_fence);
        }
        Ok(Self {
            device_manager,
            image_available_semaphores,
            render_finished_semaphores,
            in_flight_fences,
            n_pass: 0,
        })
    }

    fn wait_for_last_fence(&self) -> VkResult<()> {
        let (fence, _, _) = self.get_last_resources();
        unsafe {
            self.device_manager
                .device
                .clone()
                .wait_for_fences(&[fence], true, u64::MAX)
        }
    }

    fn wait_for_fence(&self) -> VkResult<()> {
        let (fence, _, _) = self.get_resources();
        unsafe {
            self.device_manager
                .device
                .clone()
                .wait_for_fences(&[fence], true, u64::MAX)
        }
    }

    fn get_resources(&self) -> (vk::Fence, vk::Semaphore, vk::Semaphore) {
        (
            self.in_flight_fences[self.n_pass],
            self.image_available_semaphores[self.n_pass],
            self.render_finished_semaphores[self.n_pass],
        )
    }

    fn get_last_resources(&self) -> (vk::Fence, vk::Semaphore, vk::Semaphore) {
        let n_pass = (self.n_pass + 1) % 2;
        (
            self.in_flight_fences[n_pass],
            self.image_available_semaphores[n_pass],
            self.render_finished_semaphores[n_pass],
        )
    }

    fn reset_fence(&self) -> VkResult<()> {
        let fence = self.in_flight_fences[self.n_pass];

        if !fence.is_null() {
            unsafe { self.device_manager.device.clone().reset_fences(&[fence]) }
        } else {
            Ok(())
        }
    }

    fn next_pass(&mut self) {
        self.n_pass = (self.n_pass + 1) % 2;
    }
}

pub struct GpuFuture {
    device_manager: Arc<DeviceManager>,
    command_buffers: Mutex<Vec<Arc<CommandBuffer>>>,
    queue: Arc<Queue>,
    sync: Arc<Mutex<GpuSync>>,
}

impl Drop for GpuFuture {
    fn drop(&mut self) {
        if Arc::strong_count(&self.sync) != 1 {
            return;
        }
        let lock = self.sync.lock().unwrap();

        unsafe {
            self.queue.wait_idle().unwrap();

            lock.image_available_semaphores
                .iter()
                .chain(lock.render_finished_semaphores.iter())
                .for_each(|&semaphore| {
                    self.device_manager
                        .device
                        .destroy_semaphore(semaphore, None)
                });

            lock.in_flight_fences
                .iter()
                .for_each(|&fence| self.device_manager.device.destroy_fence(fence, None));
        }
    }
}

unsafe impl Send for GpuFuture {}
unsafe impl Sync for GpuFuture {}

impl GpuFuture {
    fn from_command_entry(
        device_manager: Arc<DeviceManager>,
        buffers: Vec<Arc<CommandBuffer>>,
        queue: Arc<Queue>,
    ) -> Box<Self> {
        Box::new(Self {
            device_manager: device_manager.clone(),
            command_buffers: Mutex::new(buffers),
            sync: Arc::new(Mutex::new(GpuSync::new(device_manager).unwrap())),
            queue,
        })
    }

    pub fn n_pass(self: &Self) -> usize {
        self.sync.lock().unwrap().n_pass
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

    pub fn wait(self: Box<Self>) -> VkResult<Box<Self>> {
        self.sync.lock().unwrap().wait_for_last_fence()?;

        Ok(self)
    }

    pub fn acquire_next_image(&self, presentation: &Presentation) -> VkResult<(u32, bool)> {
        let sync = self.sync.lock().unwrap();

        let (_, image_available_semaphore, _) = sync.get_resources();

        let result;

        unsafe {
            result = presentation
                .swapchain
                .swapchain
                .write()
                .unwrap()
                .acquire_next_image(
                    *presentation.swapchain.swapchain_khr.read().unwrap(),
                    u64::MAX,
                    image_available_semaphore,
                    vk::Fence::null(),
                );
        }

        sync.reset_fence().unwrap();

        result
    }

    pub fn then_swapchain_present_and_flush(
        self: Box<Self>,
        presentation: Arc<Presentation>,
    ) -> Result<Box<Self>, (vk::Result, Arc<Mutex<GpuSync>>)> {
        let swapchain = presentation.swapchain.clone();

        let mut sync = self.sync.lock().unwrap();
        let (fence, image_available_semaphore, render_finished_semaphore) = sync.get_resources();

        let image_semaphores = [image_available_semaphore];
        let render_semaphores = [render_finished_semaphore];

        let swaphchains = [*swapchain.swapchain_khr.read().unwrap()];
        let indices = [*presentation.image_index.lock().unwrap()];
        let mut command_buffers_lock = self.command_buffers.lock().unwrap();

        let command_buffers: Vec<vk::CommandBuffer> = command_buffers_lock
            .iter()
            .map(|cb| cb.handler)
            .filter(|h| !h.is_null())
            .collect();

        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(&image_semaphores)
            .signal_semaphores(&render_semaphores)
            .wait_dst_stage_mask(&[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
            .command_buffers(&command_buffers);

        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&render_semaphores)
            .swapchains(&swaphchains)
            .image_indices(&indices);

        let result;

        let queue_lock = self.queue.submit_still_lock(&[submit_info], fence).unwrap();

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

        if result.is_ok() {
            sync.next_pass();
        } else {
            return Err((result.err().unwrap(), self.sync.clone()));
        }

        drop(sync);

        Ok(self)
    }

    pub fn flush(self: Box<Self>) -> CrystalResult<Box<Self>> {
        let mut command_buffer_lock = self.command_buffers.lock().unwrap();
        let command_buffers: Vec<vk::CommandBuffer> = (*command_buffer_lock)
            .iter()
            .map(|buffer| buffer.handler)
            .collect();
        let submit_info = vk::SubmitInfo::default().command_buffers(&command_buffers);

        self.queue
            .submit(&[submit_info], vk::Fence::null())
            .unwrap();

        drop(command_buffers);
        command_buffer_lock.clear();
        drop(command_buffer_lock);

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
    queue: Arc<Queue>,
    pub double_buffering: bool,
}

impl CommandEntry {
    pub fn now_with_sync(&self, sync: Arc<Mutex<GpuSync>>) -> Box<GpuFuture> {
        Box::new(GpuFuture {
            device_manager: self.device_manager.clone(),
            command_buffers: Mutex::new(vec![]),
            sync,
            queue: self.queue.clone(),
        })
    }

    pub fn now(&self) -> Box<GpuFuture> {
        GpuFuture::from_command_entry(self.device_manager.clone(), vec![], self.queue.clone())
    }

    fn new(
        device_manager: Arc<DeviceManager>,
        queue: Arc<Queue>,
        double_buffering: bool,
    ) -> CrystalResult<Self> {
        let command_pool = CommandPool::new(device_manager.clone(), queue.clone())?;

        let command_buffers = CommandBuffer::new(
            command_pool.clone(),
            if double_buffering { 2 } else { 1 },
            vk::CommandBufferLevel::PRIMARY,
        )?;

        Ok(Self {
            device_manager,
            command_pool,
            command_buffers,
            queue,
            double_buffering,
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

        Ok(GpuFuture::from_command_entry(
            self.device_manager.clone(),
            command_buffer_managers,
            self.queue.clone(),
        ))
    }

    pub fn record_command_buffer<P>(
        &self,
        n_pass: usize,
        predicate: P,
    ) -> CrystalResult<Box<GpuFuture>>
    where
        P: Fn(&vk::CommandBuffer, Arc<ash::Device>),
    {
        let n_pass = if self.double_buffering { n_pass } else { 0 };

        match unsafe {
            self.device_manager.device.reset_command_buffer(
                self.command_buffers[n_pass].handler,
                vk::CommandBufferResetFlags::empty(),
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

        Ok(GpuFuture::from_command_entry(
            self.device_manager.clone(),
            vec![self.command_buffers[n_pass].clone()],
            self.queue.clone(),
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
        double_buffering: bool,
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
                    double_buffering,
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
                    double_buffering,
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
                    double_buffering,
                )?),
            );
        }

        Ok(Arc::new(Self {
            device_manager,
            command_entries,
        }))
    }
}
