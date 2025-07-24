use std::ffi::c_void;

use ash::vk::{
    self, DebugUtilsMessageSeverityFlagsEXT, DebugUtilsMessageTypeFlagsEXT,
    DebugUtilsMessengerCallbackDataEXT,
};

use ash::ext::debug_utils;
use ash::{Entry, Instance};

use crate::debug::log;
use crate::errors::{GraphicsError, GraphicsResult};

pub(crate) struct DebugUtilsMessanger {
    _debug_utils: debug_utils::Instance,
    _debug_utils_messanger: vk::DebugUtilsMessengerEXT,
}

impl Drop for DebugUtilsMessanger {
    fn drop(&mut self) {
        unsafe {
            self._debug_utils
                .destroy_debug_utils_messenger(self._debug_utils_messanger, None);
        }
    }
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
            } else {
                log!("DEBUG: {}", cstr.to_str().unwrap());
            }
        }
        None => log!("debug callback was called, but invalid callback data was provided"),
    }
    0
}

pub(crate) fn create_debug_utils_messanger(
    entry: &Entry,
    instance: &Instance,
) -> GraphicsResult<DebugUtilsMessanger> {
    let debug_utils = debug_utils::Instance::new(entry, instance);
    let debug_messanger_create_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
        .message_severity(
            DebugUtilsMessageSeverityFlagsEXT::ERROR
                | DebugUtilsMessageSeverityFlagsEXT::WARNING
                | DebugUtilsMessageSeverityFlagsEXT::VERBOSE, // | DebugUtilsMessageSeverityFlagsEXT::INFO,
        )
        .message_type(
            DebugUtilsMessageTypeFlagsEXT::GENERAL
                | DebugUtilsMessageTypeFlagsEXT::VALIDATION
                | DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
        )
        .pfn_user_callback(Some(debug_callback));

    let debug_utils_messanger = match unsafe {
        debug_utils.create_debug_utils_messenger(&debug_messanger_create_info, None)
    } {
        Ok(messanger) => messanger,
        Err(e) => {
            log!("cannot create vulkan debug messanger: {}", e);
            return Err(GraphicsError::DebugError);
        }
    };

    Ok(DebugUtilsMessanger {
        _debug_utils: debug_utils,
        _debug_utils_messanger: debug_utils_messanger,
    })
}
