#![warn(missing_docs)]
#![warn(unreachable_pub)]

//! # Crystal API
//! Crystal API is a unified wrapper for GPU APIs designed for the best capability
//! with any solutions in apps development

/// Debug module
pub mod debug;
/// Errors module
pub mod errors;
mod gpu_data;
/// Mesh module
pub mod mesh;
/// Object module
pub mod object;
/// Settings module
pub mod settings;
mod shader;
mod traits;
mod vulkan;

use std::sync::Arc;

pub use gpu_data::*;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
pub use settings::GraphicsApiInitSettings;
pub use shader::{Shader, ShaderStage};
pub use traits::*;

use crate::{errors::GraphicsResult, vulkan::VulkanEntry};

/// Creates api instance with presentation support
/// ```rust
/// let graphics = init_api_instance_with_presentation(&self.settings, &window)
///     .expect("cannot create entry");
/// ```
pub fn init_api_instance_with_presentation<T: HasWindowHandle + HasDisplayHandle>(
    settings: &GraphicsApiInitSettings,
    window: &T,
) -> GraphicsResult<Arc<dyn traits::GraphicsApi>> {
    VulkanEntry::with_presentation(settings, window)
}

/// Creates api instance for compute operations
/// ```rust
/// let graphics = init_api_instance().expect("cannot create entry");
/// ```
pub fn init_api_instance() -> GraphicsResult<Arc<dyn traits::GraphicsApi>> {
    VulkanEntry::no_presentation()
}
