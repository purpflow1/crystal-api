use std::ffi::c_void;

use ash::vk::{
    self, DebugUtilsMessageSeverityFlagsEXT, DebugUtilsMessageTypeFlagsEXT,
    DebugUtilsMessengerCallbackDataEXT,
};

use ash::ext::debug_utils;
use ash::{Entry, Instance};

use crate::debug::log;
use crate::errors::{CrystalError, CrystalResult};

pub struct DebugUtilsMessanger {
    _debug_utils: debug_utils::Instance,
    _debug_utils_messanger: vk::DebugUtilsMessengerEXT,
}

#[allow(unused_variables)]
unsafe extern "system" fn debug_callback(
    message_severity: DebugUtilsMessageSeverityFlagsEXT,
    message_type: DebugUtilsMessageTypeFlagsEXT,
    callback_data_ptr: *const DebugUtilsMessengerCallbackDataEXT,
    user_data_ptr: *mut c_void,
) -> u32 {
    let callback_data = unsafe { callback_data_ptr.read() };
    match unsafe { callback_data.message_as_c_str() } {
        Some(cstr) => {
            if message_severity.contains(DebugUtilsMessageSeverityFlagsEXT::ERROR) {
                panic!("fatal: {}", cstr.to_str().unwrap());
            } else if message_severity.contains(DebugUtilsMessageSeverityFlagsEXT::INFO) {
                log!("INFO: {}", cstr.to_str().unwrap());
            } else {
                log!("DEBUG_CALLBACK: {}", cstr.to_str().unwrap());
            }
        }
        None => log!("debug callback was called, but invalid callback data was provided"),
    }
    0
}

pub fn create_debug_utils_messanger(
    entry: &Entry,
    instance: &Instance,
) -> CrystalResult<DebugUtilsMessanger> {
    let debug_utils = debug_utils::Instance::new(entry, instance);
    let debug_messanger_create_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
        .message_severity(
            DebugUtilsMessageSeverityFlagsEXT::ERROR
                | DebugUtilsMessageSeverityFlagsEXT::WARNING
                | DebugUtilsMessageSeverityFlagsEXT::VERBOSE
                | DebugUtilsMessageSeverityFlagsEXT::INFO,
        )
        .message_type(
            DebugUtilsMessageTypeFlagsEXT::GENERAL
                | DebugUtilsMessageTypeFlagsEXT::VALIDATION
                | DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                | DebugUtilsMessageTypeFlagsEXT::DEVICE_ADDRESS_BINDING,
        )
        .pfn_user_callback(Some(debug_callback));

    let debug_utils_messanger = match unsafe {
        debug_utils.create_debug_utils_messenger(&debug_messanger_create_info, None)
    } {
        Ok(messanger) => messanger,
        Err(e) => {
            log!("cannot create vulkan debug messanger: {}", e);
            return Err(CrystalError::CannotCreateDebugMessanger);
        }
    };

    Ok(DebugUtilsMessanger {
        _debug_utils: debug_utils,
        _debug_utils_messanger: debug_utils_messanger,
    })
}
