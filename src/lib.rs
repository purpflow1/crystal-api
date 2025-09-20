#![deny(clippy::panic)]
#![deny(clippy::expect_used)]
#![deny(clippy::panic_in_result_fn)]
#![warn(missing_docs)]
#![warn(unreachable_pub)]

//! # Crystal API
//! Crystal API is a unified wrapper for GPU APIs designed for the best capability
//! with any solutions in apps development

#[allow(missing_docs)]
pub mod bitflags;
mod buffer;
#[allow(missing_docs)]
pub mod debug;
mod device;
#[allow(missing_docs)]
pub mod errors;
mod gpu_sampler_set;
mod layout;
mod mesh;
mod object;
mod pipeline;
mod proxies;
mod render_target;
mod settings;
mod shader;
#[cfg(all(test, debug_assertions))]
mod tests;
mod texture;
mod vulkan;

pub use buffer::Buffer;
pub use device::Device;
pub use mesh::*;
pub use object::Object;
pub use render_target::RenderTarget;
pub use texture::Texture;

pub use gpu_sampler_set::*;
pub use settings::GraphicsApiInitSettings;
pub use shader::{Shader, ShaderStage};
