use std::sync::{Arc, RwLock};

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

pub struct GpuFuture {
    device_manager: Arc<DeviceManager>,
    command_buffer: Arc<RwLock<CommandBufferManager>>,
    fence: RwLock<vk::Fence>,
}

unsafe impl Send for GpuFuture {}
unsafe impl Sync for GpuFuture {}

impl GpuFuture {
    pub fn now(device_manager: Arc<DeviceManager>) -> Arc<Self> {
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
    ) -> Arc<Self> {
        let buffer = (*buffer).clone();

        Arc::new(Self {
            device_manager,
            command_buffer: Arc::new(RwLock::new(buffer)),
            fence: RwLock::new(vk::Fence::null()),
        })
    }

    pub fn join(self: Arc<Self>, other: Arc<Self>) -> Arc<Self> {
        *self.command_buffer.write().unwrap() = other.command_buffer.read().unwrap().clone();
        self
    }

    pub fn wait(self: Arc<Self>) -> VkResult<Arc<Self>> {
        let fence = *self.fence.read().unwrap();

        unsafe {
            self.device_manager
                .device
                .clone()
                .wait_for_fences(&[fence], true, u64::MAX)?
        };

        Ok(self)
    }

    pub fn acquire_next_image(
        self: Arc<Self>,
        presentation: &Presentation,
    ) -> VkResult<(u32, bool)> {
        unsafe {
            let result = presentation
                .swapchain
                .swapchain
                .write()
                .unwrap()
                .acquire_next_image(
                    *presentation.swapchain.swapchain_khr.read().unwrap(),
                    u64::MAX,
                    presentation.image_available_semaphores[presentation.current_frame],
                    vk::Fence::null(),
                );

            let next_fence = presentation.in_flight_fences[presentation.current_frame];
            *self.fence.write().unwrap() = next_fence;

            if !next_fence.is_null() {
                self.device_manager
                    .device
                    .clone()
                    .reset_fences(&[next_fence])?;
            }

            result
        }
    }

    pub fn then_swapchain_present(
        self: Arc<Self>,
        command_entry: Arc<CommandEntry>,
        presentation: &mut Presentation,
    ) -> VkResult<Arc<GpuFuture>> {
        let swapchain = presentation.swapchain.clone();
        let wait_semaphores = [presentation.image_available_semaphores[presentation.current_frame]];
        let signal_semaphores =
            [presentation.render_finished_semaphores[presentation.current_frame]];

        let fence = presentation.in_flight_fences[presentation.current_frame];
        *self.fence.write().unwrap() = fence;

        let swaphchains = [*swapchain.swapchain_khr.read().unwrap()];
        let indices = [presentation.image_index];
        let command_buffer = self.command_buffer.clone();
        let device_manager = self.device_manager.clone();

        let command_buffers = [command_buffer.read().unwrap().handler];

        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(&wait_semaphores)
            .signal_semaphores(&signal_semaphores)
            .wait_dst_stage_mask(&[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
            .command_buffers(&command_buffers);

        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&signal_semaphores)
            .swapchains(&swaphchains)
            .image_indices(&indices);

        unsafe {
            device_manager
                .device
                .queue_submit(command_entry.queue, &[submit_info], fence)?;
            swapchain
                .swapchain
                .write()
                .unwrap()
                .queue_present(command_entry.queue, &present_info)?
        };

        presentation.current_frame =
            (presentation.current_frame + 1) % presentation.frames_in_flight as usize;

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
    queue: vk::Queue,
    queue_family_index: u32,
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
        command_buffer_count: u32,
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
            .command_buffer_count(command_buffer_count);

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
        buffer_idx: usize,
        predicate: P,
    ) -> CrystalResult<Arc<GpuFuture>>
    where
        P: Fn(&vk::CommandBuffer, Arc<ash::Device>),
    {
        match unsafe {
            self.device_manager.device.reset_command_buffer(
                self.command_buffers[buffer_idx].handler,
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
                .begin_command_buffer(self.command_buffers[buffer_idx].handler, &begin_info)
        } {
            Ok(_) => {}
            Err(e) => {
                log!("cannot begin command buffer: {}", e);
                return Err(CrystalError::CommandManagerError);
            }
        };

        predicate(
            &self.command_buffers[buffer_idx].handler,
            self.device_manager.device.clone(),
        );

        match unsafe {
            self.device_manager
                .device
                .end_command_buffer(self.command_buffers[buffer_idx].handler)
        } {
            Ok(_) => {}
            Err(e) => {
                log!("cannot end command buffer: {}", e);
                return Err(CrystalError::CommandManagerError);
            }
        };

        Ok(GpuFuture::from_command_entry(
            self.device_manager.clone(),
            self.command_buffers[buffer_idx].clone(),
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
        graphics_command_buffer_count: u32,
    ) -> CrystalResult<Arc<Self>> {
        let graphics = match device_manager.queue_families_indices.graphics_index {
            Some(idx) => Some(Arc::new(CommandEntry::new(
                device_manager.clone(),
                idx,
                vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
                graphics_command_buffer_count,
            )?)),
            None => None,
        };

        let present = match device_manager.queue_families_indices.present_index {
            Some(idx) => Some(Arc::new(CommandEntry::new(
                device_manager.clone(),
                idx,
                vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
                graphics_command_buffer_count,
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
