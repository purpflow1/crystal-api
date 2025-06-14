use std::sync::Arc;

use smallvec::smallvec;
use vulkano::{
    buffer::BufferUsage,
    command_buffer::{BlitImageInfo, CopyBufferToImageInfo, ImageBlit},
    device::DeviceOwned,
    format::{Format, FormatFeatures},
    image::{
        ImageAspects, ImageCreateInfo, ImageLayout, ImageSubresourceLayers, ImageSubresourceRange,
        ImageTiling, ImageType, ImageUsage, SampleCount,
        sampler::Filter,
        view::{ImageView, ImageViewCreateInfo, ImageViewType},
    },
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    sync::{self, GpuFuture},
};

use crate::{
    debug::log,
    errors::{CrystalError, CrystalResult},
    images::Image2D,
    traits,
    vulkan::commands::CommandEntry,
};

use super::{commands::CommandManager, memory::BufferManager};

pub struct Image {
    memory_allocator: Arc<StandardMemoryAllocator>,
    pub image_view: Arc<ImageView>,
    pub layout: ImageLayout,
    pub extent: [u32; 3],
    pub mip_levels: u32,
    pub anisotropy_texels: f32,
}

impl Image {
    pub(crate) fn new(
        memory_allocator: Arc<StandardMemoryAllocator>,
        extent: [u32; 2],
        samples: SampleCount,
        format: Format,
        tiling: ImageTiling,
        aspect_mask: ImageAspects,
        usage: ImageUsage,
        generate_mips: bool,
        anisotropy_texels: f32,
    ) -> CrystalResult<Self> {
        let layout = ImageLayout::Undefined;

        let extent = [extent[0], extent[1], 1];

        let mip_levels = if generate_mips {
            (extent[0] as f32).max(extent[1] as f32).log2().floor() as u32
        } else {
            1
        };

        let create_info = ImageCreateInfo {
            image_type: ImageType::Dim2d,
            extent,
            mip_levels,
            array_layers: 1,
            format,
            tiling,
            initial_layout: layout,
            usage,
            samples,
            ..Default::default()
        };

        let device = memory_allocator.device();

        let memory_requirements = match device.image_memory_requirements(create_info.clone(), None)
        {
            Ok(req) => req,
            Err(e) => {
                log!("cannot get image memory requirements: {:?}", e);
                return Err(CrystalError::ImageError);
            }
        };

        let allocation_create_info = AllocationCreateInfo {
            memory_type_bits: memory_requirements.memory_type_bits,
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        };

        let image = match vulkano::image::Image::new(
            memory_allocator.clone(),
            create_info,
            allocation_create_info,
        ) {
            Ok(image) => image,
            Err(e) => {
                log!("cannot create image: {:?}", e);
                return Err(CrystalError::ImageError);
            }
        };

        let create_info = ImageViewCreateInfo {
            view_type: ImageViewType::Dim2d,
            format,
            subresource_range: ImageSubresourceRange {
                aspects: aspect_mask,
                mip_levels: 0..mip_levels,
                array_layers: 0..1,
            },
            ..Default::default()
        };

        let image_view = match ImageView::new(image.clone(), create_info) {
            Ok(image_view) => image_view,
            Err(e) => {
                log!("cannot create image view: {:?}", e);
                return Err(CrystalError::ImageError);
            }
        };

        Ok(Self {
            memory_allocator,
            image_view,
            layout,
            extent,
            mip_levels,
            anisotropy_texels,
        })
    }
}

pub struct VulkanTexture {
    pub staging_buffer_manager: BufferManager<u8>,
    pub image: Image,
}

impl traits::Texture for VulkanTexture {
    fn as_vulkan_arc(self: Arc<Self>) -> Option<Arc<super::VulkanTexture>> {
        Some(self)
    }
}

impl VulkanTexture {
    pub(crate) fn new(
        memory_allocator: Arc<StandardMemoryAllocator>,
        image: &Image2D,
        anisotropy_texels: f32,
    ) -> CrystalResult<Self> {
        let buffer_manager = BufferManager::new(
            memory_allocator.clone(),
            image.pixels.clone(),
            BufferUsage::TRANSFER_SRC,
            MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
        )?;

        let format = match image.channels {
            1 => Format::R8_SRGB,
            2 => Format::R8G8_SRGB,
            3 => Format::R8G8B8_SRGB,
            _ => Format::R8G8B8A8_SRGB,
        };

        match memory_allocator
            .device()
            .physical_device()
            .format_properties(format)
        {
            Err(e) => {
                log!("fatal: cannot get format properties: {:?}", e);
                return Err(CrystalError::ImageError);
            }
            Ok(props) => {
                if !props
                    .optimal_tiling_features
                    .intersects(FormatFeatures::SAMPLED_IMAGE_FILTER_LINEAR)
                {
                    panic!(
                        "fatal: no suitable device for image linear filtering with format: {:?}",
                        format
                    );
                }
            }
        };

        let image = Image::new(
            memory_allocator.clone(),
            [image.width, image.height],
            SampleCount::Sample1,
            format,
            ImageTiling::Optimal,
            ImageAspects::COLOR,
            ImageUsage::TRANSFER_SRC | ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
            true,
            anisotropy_texels,
        )?;

        Ok(Self {
            staging_buffer_manager: buffer_manager,
            image,
        })
    }

    pub fn prepare_texture_image(&mut self, command_manager: &CommandManager) -> CrystalResult<()> {
        let command_entry = command_manager.graphics.as_ref().unwrap();

        let future = sync::now(self.image.memory_allocator.device().clone());

        let future = self.stage_image(future.boxed(), command_entry)?;
        let future = self.generate_mipmaps(future.boxed(), command_entry)?;

        future
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();

        Ok(())
    }

    fn stage_image(
        &self,
        future: Box<dyn GpuFuture>,
        command_entry: &CommandEntry,
    ) -> CrystalResult<Box<dyn GpuFuture>> {
        let command_buffer = command_entry.record_command_buffer(|command_buffer_builder| {
            let src_buffer = self.staging_buffer_manager.buffer.clone();
            let copy_buffer_to_image_info = CopyBufferToImageInfo::buffer_image(
                (*src_buffer).clone(),
                self.image.image_view.image().clone(),
            );
            command_buffer_builder
                .copy_buffer_to_image(copy_buffer_to_image_info)
                .unwrap();
        });

        let future = future
            .then_execute(command_entry.queue.clone(), command_buffer.unwrap())
            .unwrap();

        Ok(future.boxed())
    }

    fn generate_mipmaps(
        &self,
        future: Box<dyn GpuFuture>,
        command_entry: &CommandEntry,
    ) -> CrystalResult<Box<dyn GpuFuture>> {
        let command_buffer = command_entry.record_command_buffer(|command_buffer_builder| {
            let mut mip_width = self.image.extent[0];
            let mut mip_heigth = self.image.extent[1];

            for mip_level in 1..self.image.mip_levels {
                let blit = ImageBlit {
                    src_offsets: [[0, 0, 0], [mip_width, mip_heigth, 1]],
                    dst_offsets: [
                        [0, 0, 0],
                        [
                            if mip_width > 1 { mip_width / 2 } else { 1 },
                            if mip_heigth > 1 { mip_heigth / 2 } else { 1 },
                            1,
                        ],
                    ],
                    src_subresource: ImageSubresourceLayers {
                        aspects: ImageAspects::COLOR,
                        mip_level: mip_level - 1,
                        array_layers: 0..1,
                    },
                    dst_subresource: ImageSubresourceLayers {
                        aspects: ImageAspects::COLOR,
                        mip_level: mip_level,
                        array_layers: 0..1,
                    },
                    ..Default::default()
                };

                let mut blit_image_info = BlitImageInfo::images(
                    self.image.image_view.image().clone(),
                    self.image.image_view.image().clone(),
                );

                blit_image_info.regions = smallvec![blit];
                blit_image_info.filter = Filter::Linear;

                command_buffer_builder.blit_image(blit_image_info).unwrap();

                if mip_width > 1 {
                    mip_width /= 2
                }

                if mip_heigth > 1 {
                    mip_heigth /= 2
                }
            }
        });

        let future = future
            .then_execute(command_entry.queue.clone(), command_buffer.unwrap())
            .unwrap();

        Ok(future.boxed())
    }
}
