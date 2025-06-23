use std::sync::{Arc, Mutex, RwLock};

use ash::{
    prelude::VkResult,
    vk::{self, Handle},
};

use crate::{
    debug::log,
    errors::{CrystalError, CrystalResult},
    vulkan::{devices::DeviceManager, presentation::Presentation},
};

use super::images::Image;

pub struct GpuSync {
    device_manager: Arc<DeviceManager>,

    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,

    pub n_pass: usize,
}

impl Drop for GpuSync {
    fn drop(&mut self) {
        unsafe {
            self.device_manager.device.device_wait_idle().unwrap();

            self.image_available_semaphores
                .iter()
                .chain(self.render_finished_semaphores.iter())
                .for_each(|&semaphore| {
                    self.device_manager
                        .device
                        .destroy_semaphore(semaphore, None)
                });

            self.in_flight_fences
                .iter()
                .for_each(|&fence| self.device_manager.device.destroy_fence(fence, None));
        }
    }
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
    command_buffer: Arc<RwLock<CommandBufferManager>>,
    sync: Arc<Mutex<GpuSync>>,
}

unsafe impl Send for GpuFuture {}
unsafe impl Sync for GpuFuture {}

impl GpuFuture {
    pub fn now_with_sync(
        device_manager: Arc<DeviceManager>,
        sync: Arc<Mutex<GpuSync>>,
    ) -> Box<Self> {
        Box::new(Self {
            device_manager: device_manager.clone(),
            command_buffer: Arc::new(RwLock::new(CommandBufferManager {
                handler: vk::CommandBuffer::null(),
            })),
            sync,
        })
    }

    pub fn now(device_manager: Arc<DeviceManager>) -> Box<Self> {
        Self::from_command_entry(
            device_manager,
            Arc::new(CommandBufferManager {
                handler: vk::CommandBuffer::null(),
            }),
        )
    }

    fn from_command_entry(
        device_manager: Arc<DeviceManager>,
        buffer: Arc<CommandBufferManager>,
    ) -> Box<Self> {
        let buffer = (*buffer).clone();

        Box::new(Self {
            device_manager: device_manager.clone(),
            command_buffer: Arc::new(RwLock::new(buffer)),
            sync: Arc::new(Mutex::new(GpuSync::new(device_manager).unwrap())),
        })
    }

    pub fn n_pass(self: &Self) -> usize {
        self.sync.lock().unwrap().n_pass
    }

    pub fn join(self: Box<Self>, other: Box<Self>) -> Box<Self> {
        *self.command_buffer.write().unwrap() = other.command_buffer.read().unwrap().clone();
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

    pub fn then_swapchain_present(
        self: Box<Self>,
        command_entry: Arc<CommandEntry>,
        presentation: Arc<Presentation>,
    ) -> Result<Box<Self>, (vk::Result, Arc<Mutex<GpuSync>>)> {
        let swapchain = presentation.swapchain.clone();

        let mut sync = self.sync.lock().unwrap();
        let (fence, image_available_semaphore, render_finished_semaphore) = sync.get_resources();

        let image_semaphores = [image_available_semaphore];
        let render_semaphores = [render_finished_semaphore];

        let swaphchains = [*swapchain.swapchain_khr.read().unwrap()];
        let indices = [*presentation.image_index.lock().unwrap()];
        let command_buffer = self.command_buffer.clone();
        let device_manager = self.device_manager.clone();

        let command_buffers = [command_buffer.read().unwrap().handler];

        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(&image_semaphores)
            .signal_semaphores(&render_semaphores)
            .wait_dst_stage_mask(&[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
            .command_buffers(&command_buffers);

        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&render_semaphores)
            .swapchains(&swaphchains)
            .image_indices(&indices);

        let command_entry = command_entry.clone();

        let result;

        unsafe {
            device_manager
                .device
                .queue_submit(command_entry.queue, &[submit_info], fence)
                .unwrap();

            result = swapchain
                .swapchain
                .write()
                .unwrap()
                .queue_present(command_entry.queue, &present_info);
        };

        if result.is_ok() {
            sync.next_pass();
        } else {
            return Err((result.err().unwrap(), self.sync.clone()));
        }

        drop(sync);

        Ok(self)
    }
}

#[derive(Clone)]
pub struct CommandBufferManager {
    handler: vk::CommandBuffer,
}

impl CommandBufferManager {
    fn handler(&self) -> vk::CommandBuffer {
        self.handler
    }

    fn from_handlers(handlers: impl IntoIterator<Item = vk::CommandBuffer>) -> Vec<Arc<Self>> {
        handlers
            .into_iter()
            .map(|handler| Arc::new(Self { handler }))
            .collect()
    }
}

pub struct CommandEntry {
    device_manager: Arc<DeviceManager>,
    command_pool: vk::CommandPool,
    command_buffers: Vec<Arc<CommandBufferManager>>,
    pub queue: vk::Queue,
    queue_family_index: u32,
    pub double_buffering: bool,
}

impl Drop for CommandEntry {
    fn drop(&mut self) {
        unsafe {
            self.device_manager
                .device
                .queue_wait_idle(self.queue)
                .unwrap();

            self.device_manager.device.free_command_buffers(
                self.command_pool,
                &self
                    .command_buffers
                    .iter()
                    .map(|c| c.handler)
                    .collect::<Vec<vk::CommandBuffer>>(),
            );

            self.device_manager
                .device
                .destroy_command_pool(self.command_pool, None);
        }
    }
}

impl CommandEntry {
    fn new(
        device_manager: Arc<DeviceManager>,
        queue_family_index: u32,
        flags: vk::CommandPoolCreateFlags,
        double_buffering: bool,
    ) -> CrystalResult<Self> {
        let create_info = vk::CommandPoolCreateInfo::default()
            .flags(flags)
            .queue_family_index(queue_family_index);

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

        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(if double_buffering { 2 } else { 1 });

        let command_buffers = match unsafe {
            device_manager
                .device
                .allocate_command_buffers(&allocate_info)
        } {
            Ok(command_buffers) => CommandBufferManager::from_handlers(command_buffers),
            Err(e) => {
                log!("cannot allocate command buffers: {}", e);
                return Err(CrystalError::CannotCreateCommandManager);
            }
        };

        let queue = unsafe {
            device_manager
                .device
                .get_device_queue(queue_family_index, 0)
        };

        Ok(Self {
            device_manager,
            command_pool,
            command_buffers,
            queue,
            queue_family_index,
            double_buffering,
        })
    }

    fn begin_single_time_buffer(&self) -> CrystalResult<vk::CommandBuffer> {
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_pool(self.command_pool)
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

        Ok(command_buffer)
    }

    fn end_single_time_buffer(&self, command_buffer: vk::CommandBuffer) -> CrystalResult<()> {
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

        let commands_buffers = [command_buffer];
        let submit_info = vk::SubmitInfo::default().command_buffers(&commands_buffers);

        let queue = unsafe {
            self.device_manager
                .device
                .get_device_queue(self.queue_family_index, 0)
        };

        match unsafe {
            self.device_manager
                .device
                .queue_submit(queue, &[submit_info], vk::Fence::null())
        } {
            Ok(()) => (),
            Err(e) => {
                log!("cannot submit queue: {}", e);
                return Err(CrystalError::CommandManagerError);
            }
        }

        match unsafe { self.device_manager.device.queue_wait_idle(queue) } {
            Ok(()) => (),
            Err(e) => {
                log!("cannot wait idle queue: {}", e);
                return Err(CrystalError::CommandManagerError);
            }
        }

        Ok(())
    }

    pub(crate) fn generate_mipmaps(&self, image: Arc<Image>) -> CrystalResult<()> {
        let command_buffer = self.begin_single_time_buffer()?;

        let mut barrier = vk::ImageMemoryBarrier::default()
            .image(image.image)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_array_layer(0)
                    .layer_count(1)
                    .level_count(1),
            );

        let mut mip_width = image.extent.width;
        let mut mip_heigth = image.extent.height;

        for mip_level in 1..image.mip_levels {
            barrier.subresource_range = barrier.subresource_range.base_mip_level(mip_level - 1);
            barrier = barrier.old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL);
            barrier = barrier.new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
            barrier = barrier.src_access_mask(vk::AccessFlags::TRANSFER_WRITE);
            barrier = barrier.dst_access_mask(vk::AccessFlags::TRANSFER_READ);

            unsafe {
                self.device_manager.device.cmd_pipeline_barrier(
                    command_buffer,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[barrier],
                );
            }

            let blit = vk::ImageBlit::default()
                .src_offsets([
                    vk::Offset3D::default().x(0).y(0).z(0),
                    vk::Offset3D::default()
                        .x(mip_width as i32)
                        .y(mip_heigth as i32)
                        .z(1),
                ])
                .dst_offsets([
                    vk::Offset3D::default().x(0).y(0).z(0),
                    vk::Offset3D::default()
                        .x(if mip_width > 1 {
                            mip_width as i32 / 2
                        } else {
                            1
                        })
                        .y(if mip_heigth > 1 {
                            mip_heigth as i32 / 2
                        } else {
                            1
                        })
                        .z(1),
                ])
                .src_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .mip_level(mip_level - 1)
                        .base_array_layer(0)
                        .layer_count(1),
                )
                .dst_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .mip_level(mip_level)
                        .base_array_layer(0)
                        .layer_count(1),
                );

            unsafe {
                self.device_manager.device.cmd_blit_image(
                    command_buffer,
                    image.image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    image.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[blit],
                    vk::Filter::LINEAR,
                );
            }

            barrier = barrier.old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
            barrier = barrier.new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
            barrier = barrier.src_access_mask(vk::AccessFlags::TRANSFER_READ);
            barrier = barrier.dst_access_mask(vk::AccessFlags::SHADER_READ);

            unsafe {
                self.device_manager.device.cmd_pipeline_barrier(
                    command_buffer,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[barrier],
                );
            }

            if mip_width > 1 {
                mip_width /= 2
            }

            if mip_heigth > 1 {
                mip_heigth /= 2
            }
        }

        barrier.subresource_range = barrier
            .subresource_range
            .base_mip_level(image.mip_levels - 1);
        barrier = barrier.old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL);
        barrier = barrier.new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        barrier = barrier.src_access_mask(vk::AccessFlags::TRANSFER_WRITE);
        barrier = barrier.dst_access_mask(vk::AccessFlags::SHADER_READ);

        unsafe {
            self.device_manager.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
        }

        self.end_single_time_buffer(command_buffer)
    }

    pub(crate) fn transition_image_layout(
        &self,
        image: Arc<Image>,
        new_layout: vk::ImageLayout,
    ) -> CrystalResult<()> {
        let command_buffer = self.begin_single_time_buffer()?;

        let layout = *image.layout.read().unwrap();

        let mut barrier = vk::ImageMemoryBarrier::default()
            .old_layout(layout)
            .new_layout(new_layout)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image.image)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(image.mip_levels)
                    .base_array_layer(0)
                    .layer_count(1),
            );

        let mut src_stage = vk::PipelineStageFlags::TOP_OF_PIPE;
        let mut dst_stage = vk::PipelineStageFlags::TRANSFER;

        if layout == vk::ImageLayout::UNDEFINED
            && new_layout == vk::ImageLayout::TRANSFER_DST_OPTIMAL
        {
            barrier = barrier
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
        } else if layout == vk::ImageLayout::TRANSFER_DST_OPTIMAL
            && new_layout == vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        {
            barrier = barrier
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);

            src_stage = vk::PipelineStageFlags::TRANSFER;
            dst_stage = vk::PipelineStageFlags::FRAGMENT_SHADER;
        } else {
            panic!(
                "fatal: unsupported layout transition: {:?} -> {:?}",
                image.layout, new_layout
            );
        }

        *image.layout.write().unwrap() = new_layout;

        unsafe {
            self.device_manager.device.cmd_pipeline_barrier(
                command_buffer,
                src_stage,
                dst_stage,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            )
        };

        self.end_single_time_buffer(command_buffer)
    }

    pub fn copy_buffer_to_image(
        &self,
        image_manager: Arc<Image>,
        buffer: &vk::Buffer,
    ) -> CrystalResult<()> {
        let command_buffer = self.begin_single_time_buffer()?;

        let region = vk::BufferImageCopy::default()
            .buffer_offset(0)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .mip_level(0)
                    .base_array_layer(0)
                    .layer_count(1),
            )
            .image_offset(vk::Offset3D::default())
            .image_extent(image_manager.extent);

        unsafe {
            self.device_manager.device.cmd_copy_buffer_to_image(
                command_buffer,
                *buffer,
                image_manager.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
            );
        }

        self.end_single_time_buffer(command_buffer)
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
            self.command_buffers[n_pass].clone(),
        ))
    }
}

pub struct CommandManager {
    pub device_manager: Arc<DeviceManager>,
    pub graphics: Option<Arc<CommandEntry>>,
    pub present: Option<Arc<CommandEntry>>,
}

impl CommandManager {
    pub(crate) fn new(
        device_manager: Arc<DeviceManager>,
        double_buffering: bool,
    ) -> CrystalResult<Arc<Self>> {
        let graphics = match device_manager.queue_families_indices.graphics_index {
            Some(idx) => Some(Arc::new(CommandEntry::new(
                device_manager.clone(),
                idx,
                vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
                double_buffering,
            )?)),
            None => None,
        };

        let present = match device_manager.queue_families_indices.present_index {
            Some(idx) => Some(Arc::new(CommandEntry::new(
                device_manager.clone(),
                idx,
                vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
                double_buffering,
            )?)),
            None => None,
        };

        Ok(Arc::new(Self {
            device_manager,
            graphics,
            present,
        }))
    }
}
