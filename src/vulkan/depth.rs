use std::sync::Arc;

use vulkano::{
    format::{Format, FormatFeatures},
    image::ImageTiling,
};

use crate::debug::log;

pub fn find_depth_format(
    device: Arc<vulkano::device::Device>,
    tiling: ImageTiling,
    features: FormatFeatures,
) -> Format {
    let mut depth_format = None;

    for format in [Format::D32_SFLOAT_S8_UINT, Format::D24_UNORM_S8_UINT] {
        let properties = match device.physical_device().format_properties(format) {
            Ok(props) => props,
            Err(e) => {
                log!("cannot get physical device format properties: {:?}", e);
                break;
            }
        };

        if tiling == ImageTiling::Linear && properties.linear_tiling_features.intersects(features) {
            depth_format = Some(format);
            break;
        } else if tiling == ImageTiling::Optimal
            && properties.optimal_tiling_features.intersects(features)
        {
            depth_format = Some(format);
            break;
        }
    }

    if depth_format.is_none() {
        panic!("fatal: failed to find supported format for depth resources");
    }

    depth_format.unwrap()
}
