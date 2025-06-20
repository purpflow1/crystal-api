use std::sync::{Arc, RwLock};

use ash::vk;

use crate::{
    debug::log,
    errors::{CrystalError, CrystalResult},
    images::Image2D,
    traits,
};

use super::{commands::CommandManager, devices::DeviceManager, memory::BufferManager};

pub struct Image {
    pub device_manager: Arc<DeviceManager>,
    pub image: vk::Image,
    pub image_view: vk::ImageView,
    pub image_memory: vk::DeviceMemory,
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
        width: u32,
        height: u32,
        samples: vk::SampleCountFlags,
        format: vk::Format,
        tiling: vk::ImageTiling,
        aspect_mask: vk::ImageAspectFlags,
        usage: vk::ImageUsageFlags,
        mem_property: vk::MemoryPropertyFlags,
        generate_mips: bool,
        anisotropy_texels: f32,
    ) -> CrystalResult<Arc<Self>> {
        let layout = vk::ImageLayout::UNDEFINED;

        let extent = vk::Extent3D::default().width(width).height(height).depth(1);

        let mip_levels = if generate_mips {
            (height as f32).max(width as f32).log2().floor() as u32
        } else {
            1
        };

        let image_create_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .extent(extent)
            .mip_levels(mip_levels)
            .array_layers(1)
            .format(format)
            .tiling(tiling)
            .initial_layout(layout)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .samples(samples);

        let image = match unsafe { device_manager.device.create_image(&image_create_info, None) } {
            Ok(image) => image,
            Err(e) => {
                log!("cannot create image: {}", e);
                return Err(CrystalError::ImageError);
            }
        };

        let memory_requirements =
            unsafe { device_manager.device.get_image_memory_requirements(image) };

        let memory_allocate_info = vk::MemoryAllocateInfo::default()
            .allocation_size(memory_requirements.size)
            .memory_type_index(
                device_manager
                    .find_memory_type_index(mem_property, memory_requirements.memory_type_bits)?,
            );

        let image_memory = match unsafe {
            device_manager
                .device
                .allocate_memory(&memory_allocate_info, None)
        } {
            Ok(mem) => mem,
            Err(e) => {
                log!("cannot allocate image memory: {:?}", e);
                return Err(CrystalError::ImageError);
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
                return Err(CrystalError::ImageError);
            }
        };

        let image_view_create_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(aspect_mask)
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
                return Err(CrystalError::ImageError);
            }
        };

        Ok(Arc::new(Self {
            device_manager,
            image_memory,
            image,
            image_view,
            layout: RwLock::new(layout),
            extent,
            mip_levels,
            anisotropy_texels,
        }))
    }
}

pub struct VulkanTexture {
    pub staging_buffer_manager: Arc<BufferManager>,
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
        image: &Image2D,
        command_manager: Arc<CommandManager>,
        anisotropy_texels: f32,
    ) -> CrystalResult<Arc<Self>> {
        let image_size = (image.height * image.width * image.channels) as u64;

        let buffer_manager = BufferManager::new(
            device_manager.clone(),
            image_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        buffer_manager.single_time_write(&image.pixels, 0)?;

        let format = match image.channels {
            1 => vk::Format::R8_SRGB,
            2 => vk::Format::R8G8_SRGB,
            3 => vk::Format::R8G8B8_SRGB,
            _ => vk::Format::R8G8B8A8_SRGB,
        };

        let format_properties = unsafe {
            device_manager
                .instance
                .get_physical_device_format_properties(device_manager.physical_device, format)
        };

        if format_properties.optimal_tiling_features
            & vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR
            != vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR
        {
            panic!(
                "fatal: no suitable device for image linear filtering with format: {:?}",
                format
            );
        }

        let image = Image::new(
            device_manager.clone(),
            image.width,
            image.height,
            vk::SampleCountFlags::TYPE_1,
            format,
            vk::ImageTiling::OPTIMAL,
            vk::ImageAspectFlags::COLOR,
            vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::SAMPLED,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            true,
            anisotropy_texels,
        )?;

        let texture = Arc::new(Self {
            staging_buffer_manager: buffer_manager,
            image,
        });

        texture.prepare_texture_image(command_manager)?;

        Ok(texture)
    }

    pub fn prepare_texture_image(&self, command_manager: Arc<CommandManager>) -> CrystalResult<()> {
        let command_entry = command_manager.graphics.as_ref().unwrap();

        command_entry
            .transition_image_layout(self.image.clone(), vk::ImageLayout::TRANSFER_DST_OPTIMAL)?;
        command_entry
            .copy_buffer_to_image(self.image.clone(), &self.staging_buffer_manager.buffer)?;
        command_entry.generate_mipmaps(self.image.clone())?;
        /*command_entry
        .transition_image_layout(&mut self.image, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)?;*/

        Ok(())
    }
}
