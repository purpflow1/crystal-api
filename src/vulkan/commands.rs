use std::sync::Arc;

use ash::vk;

use crate::{
    debug::log,
    errors::{CrystalError, CrystalResult},
};

use super::{devices::QueueFamiliesIndices, images::Image};

pub struct CommandEntry {
    device: Arc<ash::Device>,
    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
    queue_family_index: u32,
}

impl CommandEntry {
    fn new(
        device: Arc<ash::Device>,
        queue_family_index: u32,
        flags: vk::CommandPoolCreateFlags,
        command_buffer_count: u32,
    ) -> CrystalResult<Self> {
        let create_info = vk::CommandPoolCreateInfo::default()
            .flags(flags)
            .queue_family_index(queue_family_index);

        let command_pool = match unsafe { device.create_command_pool(&create_info, None) } {
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

        let command_buffers = match unsafe { device.allocate_command_buffers(&allocate_info) } {
            Ok(command_buffers) => command_buffers,
            Err(e) => {
                log!("cannot allocate command buffers: {}", e);
                return Err(CrystalError::CannotCreateCommandManager);
            }
        };

        Ok(Self {
            device,
            command_pool,
            command_buffers,
            queue_family_index,
        })
    }

    fn begin_single_time_buffer(&self) -> CrystalResult<vk::CommandBuffer> {
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_pool(self.command_pool)
            .command_buffer_count(1);

        let command_buffer = match unsafe { self.device.allocate_command_buffers(&alloc_info) } {
            Ok(buffers) => buffers[0],
            Err(e) => {
                log!("cannot allocate command buffer: {}", e);
                return Err(CrystalError::CommandManagerError);
            }
        };

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        match unsafe {
            self.device
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
        match unsafe { self.device.end_command_buffer(command_buffer) } {
            Ok(()) => (),
            Err(e) => {
                log!("cannot begin command buffer: {}", e);
                return Err(CrystalError::CommandManagerError);
            }
        }

        let commands_buffers = [command_buffer];
        let submit_info = vk::SubmitInfo::default().command_buffers(&commands_buffers);

        let queue = unsafe { self.device.get_device_queue(self.queue_family_index, 0) };

        match unsafe {
            self.device
                .queue_submit(queue, &[submit_info], vk::Fence::null())
        } {
            Ok(()) => (),
            Err(e) => {
                log!("cannot submit queue: {}", e);
                return Err(CrystalError::CommandManagerError);
            }
        }

        match unsafe { self.device.queue_wait_idle(queue) } {
            Ok(()) => (),
            Err(e) => {
                log!("cannot wait idle queue: {}", e);
                return Err(CrystalError::CommandManagerError);
            }
        }

        Ok(())
    }

    pub(crate) fn generate_mipmaps(&self, image: &Image) -> CrystalResult<()> {
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
                self.device.cmd_pipeline_barrier(
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
                self.device.cmd_blit_image(
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
                self.device.cmd_pipeline_barrier(
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
            self.device.cmd_pipeline_barrier(
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
        image: &mut Image,
        new_layout: vk::ImageLayout,
    ) -> CrystalResult<()> {
        let command_buffer = self.begin_single_time_buffer()?;

        let mut barrier = vk::ImageMemoryBarrier::default()
            .old_layout(image.layout)
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

        if image.layout == vk::ImageLayout::UNDEFINED
            && new_layout == vk::ImageLayout::TRANSFER_DST_OPTIMAL
        {
            barrier = barrier
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
        } else if image.layout == vk::ImageLayout::TRANSFER_DST_OPTIMAL
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

        image.layout = new_layout;

        unsafe {
            self.device.cmd_pipeline_barrier(
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
        image_manager: &Image,
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
            self.device.cmd_copy_buffer_to_image(
                command_buffer,
                *buffer,
                image_manager.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
            );
        }

        self.end_single_time_buffer(command_buffer)
    }

    pub fn reset_command_buffer(&self, command_buffer_idx: usize) -> CrystalResult<()> {
        match unsafe {
            self.device.reset_command_buffer(
                self.command_buffers[command_buffer_idx],
                vk::CommandBufferResetFlags::empty(),
            )
        } {
            Ok(()) => {}
            Err(e) => {
                log!("failed resetting command buffer: {}", e);
                return Err(CrystalError::CommandManagerError);
            }
        };
        Ok(())
    }

    pub fn submit_command_buffer(
        &self,
        command_buffer_idx: usize,
        wait_semaphores: &[vk::Semaphore],
        signal_semaphores: &[vk::Semaphore],
        wait_stages: &[vk::PipelineStageFlags],
        fence: vk::Fence,
    ) -> CrystalResult<()> {
        let command_buffers = &[self.command_buffers[command_buffer_idx]];
        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(wait_semaphores)
            .signal_semaphores(signal_semaphores)
            .wait_dst_stage_mask(wait_stages)
            .command_buffers(command_buffers);

        let queue = unsafe { self.device.get_device_queue(self.queue_family_index, 0) };
        match unsafe { self.device.queue_submit(queue, &[submit_info], fence) } {
            Ok(()) => {}
            Err(e) => {
                log!("cannot submit command buffer: {}", e);
                return Err(CrystalError::CommandManagerError);
            }
        };

        Ok(())
    }

    pub fn record_command_buffer<P>(&self, buffer_idx: usize, predicate: P) -> CrystalResult<()>
    where
        P: Fn(&vk::CommandBuffer, Arc<ash::Device>),
    {
        let begin_info = vk::CommandBufferBeginInfo::default();

        match unsafe {
            self.device
                .begin_command_buffer(self.command_buffers[buffer_idx], &begin_info)
        } {
            Ok(_) => {}
            Err(e) => {
                log!("cannot begin command buffer: {}", e);
                return Err(CrystalError::CommandManagerError);
            }
        };

        predicate(&self.command_buffers[buffer_idx], self.device.clone());

        match unsafe {
            self.device
                .end_command_buffer(self.command_buffers[buffer_idx])
        } {
            Ok(_) => {}
            Err(e) => {
                log!("cannot end command buffer: {}", e);
                return Err(CrystalError::CommandManagerError);
            }
        };

        Ok(())
    }
}

pub struct CommandManager {
    pub graphics: Option<CommandEntry>,
}

impl CommandManager {
    pub fn new(
        device: Arc<ash::Device>,
        queue_families: &QueueFamiliesIndices,
        graphics_command_buffer_count: u32,
    ) -> CrystalResult<Self> {
        let graphics = match queue_families.graphics_index {
            Some(idx) => Some(CommandEntry::new(
                device,
                idx,
                vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
                graphics_command_buffer_count,
            )?),
            None => None,
        };

        Ok(Self { graphics })
    }
}
