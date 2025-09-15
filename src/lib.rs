#![deny(clippy::panic)]
#![deny(clippy::expect_used)]
#![deny(clippy::panic_in_result_fn)]
#![warn(missing_docs)]
#![warn(unreachable_pub)]

//! # Crystal API
//! Crystal API is a unified wrapper for GPU APIs designed for the best capability
//! with any solutions in apps development

/// Buffer module
pub mod buffer;
/// Debug module
pub mod debug;
/// Device module
pub mod device;
/// Errors module
pub mod errors;
mod gpu_sampler_set;
/// Layout module
pub mod layout;
/// Mesh module
pub mod mesh;
/// Object module
pub mod object;
/// Pipeline module
pub mod pipeline;
mod proxies;
/// Render target module
pub mod render_target;
/// Settings module
pub mod settings;
mod shader;
#[cfg(test)]
mod tests;
/// Texture module
pub mod texture;
mod vulkan;

pub use device::Device;

pub use gpu_sampler_set::*;
pub use settings::GraphicsApiInitSettings;
pub use shader::{Shader, ShaderStage};
