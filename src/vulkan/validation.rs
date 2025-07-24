use ash::Entry;

use crate::debug::log;

const VALIDATION_LAYERS: &[&str] = &["VK_LAYER_KHRONOS_validation"];

pub(crate) fn get_supported_validation_layers(entry: &Entry) -> Vec<[i8; 256]> {
    let mut supported_layers = Vec::new();

    let available_layers = match unsafe { entry.enumerate_instance_layer_properties() } {
        Ok(props) => props,
        Err(e) => {
            log!("cannot enumerate vulkan instance layer properties: {}", e);
            return vec![];
        }
    };
    for left_layer in available_layers {
        let left = left_layer.layer_name_as_c_str().unwrap().to_str().unwrap();
        for &right in VALIDATION_LAYERS {
            if left == right {
                supported_layers.push(left_layer.layer_name)
            }
        }
    }

    supported_layers
}
