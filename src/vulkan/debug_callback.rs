use std::process::abort;

use vulkano::instance::debug::{
    DebugUtilsMessageSeverity, DebugUtilsMessageType, DebugUtilsMessengerCallback,
    DebugUtilsMessengerCallbackData, DebugUtilsMessengerCreateInfo,
};

use crate::debug::log;

pub(crate) fn create_debug_utils_messanger_create_info() -> DebugUtilsMessengerCreateInfo {
    let debug_utils_messanger_callback =
        unsafe { DebugUtilsMessengerCallback::new(debug_callback) };

    let debug_utils_messanger_create_info = DebugUtilsMessengerCreateInfo {
        message_severity: DebugUtilsMessageSeverity::ERROR
            | DebugUtilsMessageSeverity::WARNING
            | DebugUtilsMessageSeverity::INFO
            | DebugUtilsMessageSeverity::VERBOSE,
        message_type: DebugUtilsMessageType::GENERAL
            | DebugUtilsMessageType::VALIDATION
            | DebugUtilsMessageType::PERFORMANCE,
        ..DebugUtilsMessengerCreateInfo::user_callback(debug_utils_messanger_callback.clone())
    };

    debug_utils_messanger_create_info
}

#[allow(unused_variables)]
pub(crate) fn debug_callback(
    message_severity: DebugUtilsMessageSeverity,
    message_type: DebugUtilsMessageType,
    callback_data: DebugUtilsMessengerCallbackData,
) {
    if message_severity.contains(DebugUtilsMessageSeverity::ERROR) {
        log!("fatal: {}", callback_data.message);
        log!("OBJECTS:");
        for object in callback_data.objects {
            log!("= {}", object.object_name.unwrap());
        }
        abort();
    } else {
        log!("DEBUG: {}", callback_data.message);
    }
}
