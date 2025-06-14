#![allow(dead_code)]

mod debug;
pub mod errors;
mod gpu_data;
pub mod images;
pub mod mesh;
pub mod object;
pub mod settings;
mod shader;
mod traits;
pub mod vulkan;

pub use gpu_data::GpuVec;
pub use settings::GraphicsApiInitSettings;
pub use shader::{Shader, ShaderStage};
pub use traits::*;

#[cfg(test)]
mod tests {

    #[test]
    fn loading_api() {}
}
