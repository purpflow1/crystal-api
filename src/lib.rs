#![warn(missing_docs)]
#![warn(unreachable_pub)]

//! # Crystal API
//! Crystal API is a unified wrapper for GPU APIs
//!
//!
mod debug;
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

pub use gpu_data::*;
pub use settings::GraphicsApiInitSettings;
pub use shader::{Shader, ShaderStage};
pub use traits::*;
pub use vulkan::VulkanEntry;
