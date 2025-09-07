use std::sync::{Arc, RwLock};

use ash::vk;

use crate::{
    debug::log,
    errors::{GraphicsError, GraphicsResult},
    traits,
};

use super::{
    commands::{CommandEntry, CommandManager, CommandType, GpuFuture},
    devices::DeviceManager,
    memory::BufferManager,
};

pub(crate) struct ImageCreateInfo {
    pub width: u32,
    pub height: u32,
    pub anisotropy_texels: f32,
    pub generate_mips: bool,

    pub samples: vk::SampleCountFlags,
    pub format: vk::Format,
    pub tiling: vk::ImageTiling,
    pub aspect_mask: vk::ImageAspectFlags,
    pub usage: vk::ImageUsageFlags,
    pub mem_property: vk::MemoryPropertyFlags,
}

pub struct Image {
    pub device_manager: Arc<DeviceManager>,
    pub image: vk::Image,
    pub image_view: vk::ImageView,
    pub image_memory: vk::DeviceMemory,
    pub format: vk::Format,
    pub layout: RwLock<vk::ImageLayout>,
    pub extent: vk::Extent3D,
    pub mip_levels: u32,
    pub anisotropy_texels: f32,
}

impl Drop for Image {
    fn drop(&mut self) {
        unsafe {
            self.device_manager
                .device
                .free_memory(self.image_memory, None);
            self.device_manager
                .device
                .destroy_image_view(self.image_view, None);
            self.device_manager.device.destroy_image(self.image, None);
        }
    }
}

impl Image {
    pub(crate) fn new(
        device_manager: Arc<DeviceManager>,
        image_create_info: ImageCreateInfo,
    ) -> GraphicsResult<Arc<Self>> {
        let extent = vk::Extent3D::default()
            .width(image_create_info.width)
            .height(image_create_info.height)
            .depth(1);

        let mip_levels = if image_create_info.generate_mips {
            (image_create_info.height as f32)
                .max(image_create_info.width as f32)
                .log2()
                .floor() as u32
        } else {
            1
        };

        let ty = vk::ImageType::TYPE_2D;

        let mut compression_control = vk::ImageCompressionControlEXT::default()
            .flags(vk::ImageCompressionFlagsEXT::FIXED_RATE_DEFAULT);

        let create_info = vk::ImageCreateInfo::default()
            .image_type(ty)
            .extent(extent)
            .mip_levels(mip_levels)
            .array_layers(1)
            .format(image_create_info.format)
            .tiling(image_create_info.tiling)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(image_create_info.usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .samples(image_create_info.samples)
            .push_next(&mut compression_control);

        let image = match unsafe { device_manager.device.create_image(&create_info, None) } {
            Ok(image) => image,
            Err(e) => {
                log!("cannot create image: {}", e);
                return Err(GraphicsError::ImageError);
            }
        };

        let memory_requirements =
            unsafe { device_manager.device.get_image_memory_requirements(image) };

        let memory_allocate_info = vk::MemoryAllocateInfo::default()
            .allocation_size(memory_requirements.size)
            .memory_type_index(device_manager.find_memory_type_index(
                image_create_info.mem_property,
                memory_requirements.memory_type_bits,
            )?);

        let image_memory = match unsafe {
            device_manager
                .device
                .allocate_memory(&memory_allocate_info, None)
        } {
            Ok(mem) => mem,
            Err(e) => {
                log!("cannot allocate image memory: {:?}", e);
                return Err(GraphicsError::ImageError);
            }
        };

        match unsafe {
            device_manager
                .device
                .bind_image_memory(image, image_memory, 0)
        } {
            Ok(_) => (),
            Err(e) => {
                log!("cannot bind image memory: {}", e);
                return Err(GraphicsError::ImageError);
            }
        };

        let image_view_create_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(image_create_info.format)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(image_create_info.aspect_mask)
                    .base_mip_level(0)
                    .level_count(mip_levels)
                    .base_array_layer(0)
                    .layer_count(1),
            );

        let image_view = match unsafe {
            device_manager
                .device
                .create_image_view(&image_view_create_info, None)
        } {
            Ok(image_view) => image_view,
            Err(e) => {
                log!("cannot create image view: {}", e);
                return Err(GraphicsError::ImageError);
            }
        };

        Ok(Arc::new(Self {
            device_manager,
            image_memory,
            format: image_create_info.format,
            image,
            image_view,
            layout: RwLock::new(vk::ImageLayout::UNDEFINED),
            extent,
            mip_levels,
            anisotropy_texels: image_create_info.anisotropy_texels,
        }))
    }
}

pub struct VulkanTexture {
    pub image: Arc<Image>,
}

impl traits::Texture for VulkanTexture {
    fn as_vulkan(self: Arc<Self>) -> Option<Arc<super::VulkanTexture>> {
        Some(self)
    }
}

impl VulkanTexture {
    pub(crate) fn new(
        device_manager: Arc<DeviceManager>,
        extent: [u32; 2],
        anisotropy_texels: f32,
    ) -> GraphicsResult<Arc<Self>> {
        let format = vk::Format::R8G8B8A8_SRGB;

        let format_properties = unsafe {
            device_manager
                .instance
                .get_physical_device_format_properties(device_manager.physical_device, format)
        };

        if format_properties.optimal_tiling_features
            & vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR
            != vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR
        {
            panic!("fatal: no suitable device for image linear filtering with format: {format:?}");
        }

        let create_info = ImageCreateInfo {
            width: extent[0],
            height: extent[1],
            generate_mips: false,
            anisotropy_texels,
            format,
            samples: vk::SampleCountFlags::TYPE_1,
            tiling: vk::ImageTiling::OPTIMAL,
            aspect_mask: vk::ImageAspectFlags::COLOR,
            usage: vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::COLOR_ATTACHMENT,
            mem_property: vk::MemoryPropertyFlags::DEVICE_LOCAL,
        };

        let image = Image::new(device_manager.clone(), create_info)?;

        let texture = Arc::new(Self { image });

        let command_entry = if let Some(queue) = device_manager
            .queues
            .iter()
            .find(|queue| queue.flags.intersects(vk::QueueFlags::TRANSFER))
        {
            CommandEntry::new(device_manager.clone(), queue.clone(), 0)?
        } else {
            panic!("fatal: no transfer queue family")
        };

        let future = texture.transition_image_layout(
            command_entry.clone(),
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        )?;

        let future = future.join(texture.transition_image_layout(
            command_entry.clone(),
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        )?);

        future.flush_transfer(command_entry.queue.clone())?;

        Ok(texture)
    }

    pub(crate) fn new_staged(
        device_manager: Arc<DeviceManager>,
        command_manager: Arc<CommandManager>,
        buffer: Arc<BufferManager>,
        extent: [u32; 2],
        anisotropy_texels: f32,
    ) -> GraphicsResult<Arc<Self>> {
        let format = vk::Format::R8G8B8A8_SRGB;

        let format_properties = unsafe {
            device_manager
                .instance
                .get_physical_device_format_properties(device_manager.physical_device, format)
        };

        if format_properties.optimal_tiling_features
            & vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR
            != vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR
        {
            panic!("fatal: no suitable device for image linear filtering with format: {format:?}");
        }

        let create_info = ImageCreateInfo {
            width: extent[0],
            height: extent[1],
            generate_mips: true,
            anisotropy_texels,
            format,
            samples: vk::SampleCountFlags::TYPE_1,
            tiling: vk::ImageTiling::OPTIMAL,
            aspect_mask: vk::ImageAspectFlags::COLOR,
            usage: vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::SAMPLED,
            mem_property: vk::MemoryPropertyFlags::DEVICE_LOCAL,
        };

        let image = Image::new(device_manager.clone(), create_info)?;

        let texture = Arc::new(Self { image });

        texture.prepare_texture_image(buffer, command_manager)?;

        Ok(texture)
    }

    fn prepare_texture_image(
        &self,
        buffer: Arc<BufferManager>,
        command_manager: Arc<CommandManager>,
    ) -> GraphicsResult<()> {
        let command_entry = command_manager
            .command_entries
            .get(&CommandType::Transfer)
            .unwrap();

        let future = self.transition_image_layout(
            command_entry.clone(),
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        )?;
        let future = future.join(self.stage_image(buffer, command_entry.clone())?);
        let future = future.join(self.generate_mipmaps(command_entry.clone())?);

        future.flush_transfer(command_entry.queue.clone())?;

        Ok(())
    }

    pub(crate) fn generate_mipmaps(
        &self,
        command_entry: Arc<CommandEntry>,
    ) -> GraphicsResult<Box<GpuFuture>> {
        command_entry.record_single_time_buffer(|command_buffer, device| {
            let mut barrier = vk::ImageMemoryBarrier::default()
                .image(self.image.image)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_array_layer(0)
                        .layer_count(1)
                        .level_count(1),
                );

            let mut mip_width = self.image.extent.width;
            let mut mip_heigth = self.image.extent.height;

            for mip_level in 1..self.image.mip_levels {
                barrier.subresource_range = barrier.subresource_range.base_mip_level(mip_level - 1);
                barrier = barrier.old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL);
                barrier = barrier.new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
                barrier = barrier.src_access_mask(vk::AccessFlags::TRANSFER_WRITE);
                barrier = barrier.dst_access_mask(vk::AccessFlags::TRANSFER_READ);

                unsafe {
                    device.cmd_pipeline_barrier(
                        *command_buffer,
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
                    device.cmd_blit_image(
                        *command_buffer,
                        self.image.image,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        self.image.image,
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
                    device.cmd_pipeline_barrier(
                        *command_buffer,
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
                .base_mip_level(self.image.mip_levels - 1);
            barrier = barrier.old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL);
            barrier = barrier.new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
            barrier = barrier.src_access_mask(vk::AccessFlags::TRANSFER_WRITE);
            barrier = barrier.dst_access_mask(vk::AccessFlags::SHADER_READ);

            unsafe {
                device.cmd_pipeline_barrier(
                    *command_buffer,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[barrier],
                );
            }
        })
    }

    pub(crate) fn stage_image(
        &self,
        buffer: Arc<BufferManager>,
        command_entry: Arc<CommandEntry>,
    ) -> GraphicsResult<Box<GpuFuture>> {
        command_entry.record_single_time_buffer(|command_buffer, device| {
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
                .image_extent(self.image.extent);

            unsafe {
                device.cmd_copy_buffer_to_image(
                    *command_buffer,
                    buffer.get_handlers()[0],
                    self.image.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[region],
                );
            }
        })
    }

    pub(crate) fn transition_image_layout(
        &self,
        command_entry: Arc<CommandEntry>,
        new_layout: vk::ImageLayout,
    ) -> GraphicsResult<Box<GpuFuture>> {
        command_entry.record_single_time_buffer(|command_buffer, device| {
            let layout = *self.image.layout.read().unwrap();

            let mut barrier = vk::ImageMemoryBarrier::default()
                .old_layout(layout)
                .new_layout(new_layout)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(self.image.image)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(0)
                        .level_count(self.image.mip_levels)
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
                    self.image.layout, new_layout
                );
            }

            *self.image.layout.write().unwrap() = new_layout;

            unsafe {
                device.cmd_pipeline_barrier(
                    *command_buffer,
                    src_stage,
                    dst_stage,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[barrier],
                )
            };
        })
    }
}
