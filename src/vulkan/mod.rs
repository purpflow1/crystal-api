mod commands;
#[cfg(debug_assertions)]
mod debug_callback;
mod depth;
mod devices;
mod images;
mod layout;
mod library;
mod memory;
mod presentation;
mod rendering;
mod sync;
#[cfg(debug_assertions)]
mod validation;

pub(crate) use images::VulkanTexture;
pub(crate) use layout::VulkanLayout;
pub(crate) use layout::VulkanPipeline;
pub(crate) use library::VulkanEntry;
pub(crate) use memory::BufferManager;
pub(crate) use rendering::VulkanRenderTarget;

pub(super) const API_VERSION_LATEST: u32 = u32::MAX;
