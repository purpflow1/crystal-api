// renderdoc.rs
use std::ffi::c_void;
use std::ptr;

use raw_window_handle::{HasRawWindowHandle, HasWindowHandle};

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct RENDERDOC_DevicePointer(pub *mut c_void);

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct RENDERDOC_WindowHandle(pub *mut c_void);

pub type RENDERDOC_Version = i32;

// RenderDoc API versions (up to 1.4.1)
pub const eRENDERDOC_API_Version_1_4_1: RENDERDOC_Version = 10401;

#[repr(C)]
pub struct RENDERDOC_API_1_4_1 {
    pub StartFrameCapture: Option<
        unsafe extern "C" fn(device: RENDERDOC_DevicePointer, wnd_handle: RENDERDOC_WindowHandle),
    >,
    pub EndFrameCapture: Option<
        unsafe extern "C" fn(device: RENDERDOC_DevicePointer, wnd_handle: RENDERDOC_WindowHandle),
    >,
    // Other API functions omitted for brevity
}

pub struct RenderDoc {
    _library: libloading::Library, // Keep loaded
    api: RENDERDOC_API_1_4_1,
    device_ptr: RENDERDOC_DevicePointer,
    wnd_handle: RENDERDOC_WindowHandle,
}

impl RenderDoc {
    pub fn start_frame_capture(&self) {
        unsafe { self.api.StartFrameCapture.unwrap()(self.device_ptr, self.wnd_handle) }
    }

    pub fn end_frame_capture(&self) {
        unsafe { self.api.EndFrameCapture.unwrap()(self.device_ptr, self.wnd_handle) }
    }

    pub fn load(device_ptr: *mut u8, window: impl HasWindowHandle) -> Option<Self> {
        let library_name = if cfg!(target_os = "windows") {
            "renderdoc.dll"
        } else if cfg!(target_os = "linux") {
            "librenderdoc.so"
        } else {
            return None;
        };

        let library = unsafe { libloading::Library::new(library_name) }.ok()?;
        let get_api: libloading::Symbol<unsafe extern "C" fn(i32, *mut *mut c_void) -> i32> =
            unsafe { library.get(b"RENDERDOC_GetAPI") }.ok()?;

        let mut api_ptr: *mut c_void = ptr::null_mut();
        let ret = unsafe {
            get_api(
                eRENDERDOC_API_Version_1_4_1,
                &mut api_ptr as *mut *mut c_void,
            )
        };

        if ret == 1 && !api_ptr.is_null() {
            let api = unsafe { (api_ptr as *const RENDERDOC_API_1_4_1).read() };
            Some(RenderDoc {
                _library: library,
                api,
                wnd_handle: RENDERDOC_WindowHandle(
                    (&window.window_handle().unwrap().as_raw()) as *const _ as *mut c_void,
                ),
                device_ptr: RENDERDOC_DevicePointer(device_ptr as *mut c_void),
            })
        } else {
            None
        }
    }
}
