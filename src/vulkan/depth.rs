use std::sync::Arc;

use ash::vk;

use crate::{
    debug::error,
    errors::{GraphicsError, GraphicsResult},
    vulkan::images::ImageCreateInfo,
};

use super::{devices::DeviceManager, images::Image};

pub(crate) fn find_depth_format(
    device_manager: Arc<DeviceManager>,
    tiling: vk::ImageTiling,
    features: vk::FormatFeatureFlags,
) -> Option<vk::Format> {
    let mut depth_format = None;

    for format in [
        vk::Format::D32_SFLOAT_S8_UINT,
        vk::Format::D24_UNORM_S8_UINT,
    ] {
        let properties = unsafe {
            device_manager
                .instance
                .get_physical_device_format_properties(device_manager.physical_device, format)
        };

        if (tiling == vk::ImageTiling::LINEAR
            && (properties.linear_tiling_features & features) == features)
            || tiling == vk::ImageTiling::OPTIMAL
                && (properties.optimal_tiling_features & features) == features
        {
            depth_format = Some(format);
            break;
        }
    }

    depth_format
}

pub(crate) struct DepthResources {
    pub image: Arc<Image>,
}

impl DepthResources {
    pub(crate) fn new(
        device_manager: Arc<DeviceManager>,
        width: u32,
        height: u32,
        samples: vk::SampleCountFlags,
    ) -> GraphicsResult<Self> {
        let tiling = vk::ImageTiling::OPTIMAL;
        let features = vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT;

        let depth_format = match find_depth_format(device_manager.clone(), tiling, features) {
            Some(format) => format,
            None => {
                error!("cannot find supported depth format");
                return Err(GraphicsError::NotSupportedDevice);
            }
        };

        let create_info = ImageCreateInfo {
            width,
            height,
            generate_mips: false,
            anisotropy_texels: 1.,
            format: depth_format,
            samples,
            tiling,
            aspect_mask: vk::ImageAspectFlags::DEPTH,
            usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            mem_property: vk::MemoryPropertyFlags::DEVICE_LOCAL,
        };

        let image = Image::new(device_manager.clone(), create_info)?;

        Ok(Self { image })
    }
}
