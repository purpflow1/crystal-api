use std::sync::Arc;

use ash::vk;

use crate::{
    debug::error,
    errors::{GraphicsError, GraphicsResult},
    vulkan::API_VERSION_LATEST,
};

pub(crate) fn create_instance(
    addition_extension: &[*const i8],
) -> GraphicsResult<(
    Arc<ash::Entry>,
    Arc<ash::Instance>,
    Option<super::debug_callback::DebugUtilsMessanger>,
)> {
    let mut instance_extensions = vec![
        #[cfg(debug_assertions)]
        vk::EXT_DEBUG_UTILS_NAME.as_ptr(),
        #[cfg(target_vendor = "apple")]
        vk::KHR_PORTABILITY_ENUMERATION_NAME.as_ptr(),
    ];

    addition_extension
        .iter()
        .for_each(|ext| instance_extensions.push(*ext));

    let entry = match unsafe { ash::Entry::load() } {
        Ok(entry) => Arc::new(entry),
        Err(e) => {
            error!("cannot load vulkan entry: {}", e);
            return GraphicsResult::Err(GraphicsError::ConnotInitLibrary);
        }
    };

    let flags = if cfg!(target_vendor = "apple") {
        vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
    } else {
        vk::InstanceCreateFlags::empty()
    };

    let app_info = vk::ApplicationInfo::default().api_version(API_VERSION_LATEST);
    #[cfg(not(debug_assertions))]
    let create_info = vk::InstanceCreateInfo::default()
        .flags(flags)
        .application_info(&app_info)
        .enabled_extension_names(&instance_extensions);

    #[cfg(debug_assertions)]
    let layers;
    #[cfg(debug_assertions)]
    let layers_pp: Vec<*const i8>;

    #[cfg(debug_assertions)]
    let create_info = {
        layers = super::validation::get_supported_validation_layers(&entry);

        if layers.is_empty() {
            error!(
                "No validation layers found!
                Vulkan SDK should be installed for proper debug.
                Visit https://vulkan.lunarg.com/"
            );

            vk::InstanceCreateInfo::default()
                .flags(flags)
                .application_info(&app_info)
                .enabled_extension_names(&instance_extensions)
        } else {
            layers_pp = layers.iter().map(|x| x.as_ptr()).collect();

            let mut create_info = vk::InstanceCreateInfo::default()
                .flags(flags)
                .application_info(&app_info)
                .enabled_extension_names(&instance_extensions);

            create_info.pp_enabled_layer_names = layers_pp.as_ptr();
            create_info.enabled_layer_count = layers_pp.len() as u32;

            create_info
        }
    };

    let instance = match unsafe { entry.create_instance(&create_info, None) } {
        Err(e) => {
            error!("cannot create vulkan instance: {}", e);
            return GraphicsResult::Err(GraphicsError::ConnotInitLibrary);
        }
        Ok(instance) => Arc::new(instance),
    };

    let debug_utils_messanger = {
        #[cfg(debug_assertions)]
        match super::debug_callback::create_debug_utils_messanger(&entry, &instance) {
            Ok(debug_utils_messanger) => {
                crate::debug::log!("debug_utils_messanger created");
                Some(debug_utils_messanger)
            }
            Err(e) => {
                error!("debug_utils_messanger creation error: {}", e);
                None
            }
        }

        #[cfg(not(debug_assertions))]
        None
    };

    Ok((entry, instance, debug_utils_messanger))
}
