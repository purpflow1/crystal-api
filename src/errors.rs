use std::fmt::{Debug, Display};

#[derive(Debug)]
pub enum CrystalError {
    CannotLoadLibrary,
    ConnotInitLibrary,
    CannotCreateDebugMessanger,
    CannotPickPhysicalDevice,
    CannotInitDevice,
    SwapChainIsNotSupported,
    SwapChainError,
    CannotCreateRenderTarget,
    CannotCreateRenderPass,
    CannotCreateFramebuffer,
    CannotCreateCommandManager,
    CommandManagerError,
    RenderingError,
    PresentationError,
    SyncError,

    ShaderError,
    MemoryError,
    DescriptorError,
    ImageError,

    OutOfDate,
}

impl Display for CrystalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("{:?}", self))
    }
}

pub type CrystalResult<T> = Result<T, CrystalError>;
