use std::sync::Arc;

use ash::vk;

use crate::errors::CrystalResult;

use super::{devices::DeviceManager, images::Image};

pub fn find_depth_format(
    device_manager: Arc<DeviceManager>,
    tiling: vk::ImageTiling,
    features: vk::FormatFeatureFlags,
) -> vk::Format {
    let mut depth_format = vk::Format::R8_SINT;

    for format in [
        vk::Format::D32_SFLOAT_S8_UINT,
        vk::Format::D24_UNORM_S8_UINT,
    ] {
        let properties = unsafe {
            device_manager
                .instance
                .get_physical_device_format_properties(device_manager.physical_device, format)
        };

        if tiling == vk::ImageTiling::LINEAR
            && (properties.linear_tiling_features & features) == features
        {
            depth_format = format;
            break;
        } else if tiling == vk::ImageTiling::OPTIMAL
            && (properties.optimal_tiling_features & features) == features
        {
            depth_format = format;
            break;
        }
    }

    if depth_format == vk::Format::R8_SINT {
        panic!("fatal: failed to find supported format for depth resources");
    }

    depth_format
}

pub struct DepthResources {
    pub image: Arc<Image>,
}

impl DepthResources {
    pub fn new(
        device_manager: Arc<DeviceManager>,
        width: u32,
        height: u32,
        samples: vk::SampleCountFlags,
    ) -> CrystalResult<Self> {
        let tiling = vk::ImageTiling::OPTIMAL;
        let features = vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT;

        let depth_format = find_depth_format(device_manager.clone(), tiling, features);

        let image = Image::new(
            device_manager.clone(),
            width,
            height,
            samples,
            depth_format,
            vk::ImageTiling::OPTIMAL,
            vk::ImageAspectFlags::DEPTH,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            false,
            1.,
        )?;

        Ok(Self { image })
    }
}
