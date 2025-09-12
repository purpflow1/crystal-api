#![deny(clippy::panic)]
#![deny(clippy::expect_used)]
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
#[cfg(test)]
mod tests;
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
pub fn init_api_instance_with_presentation<T: HasWindowHandle + HasDisplayHandle>(
    settings: &GraphicsApiInitSettings,
    window: &T,
) -> GraphicsResult<Arc<dyn traits::GraphicsApi>> {
    VulkanEntry::with_presentation(settings, window)
}

/// Creates api instance for compute operations
pub fn init_api_instance() -> GraphicsResult<Arc<dyn traits::GraphicsApi>> {
    VulkanEntry::no_presentation()
}
