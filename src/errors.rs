use std::fmt::{Debug, Display};

#[derive(Debug)]
pub enum CrystalError {
    CannotLoadLibrary,
    ConnotInitLibrary,
    CannotCreateDebugMessanger,
    Unsupported,
    CannotInitDevice,
    SwapChainIsNotSupported,
    SwapChainError,
    CannotCreateRenderTarget,
    CannotCreateRenderPass,
    CannotCreateFramebuffer,
    CannotCreateCommandManager,
    CommandManagerError,
    RenderingError,
    SyncError,
    GpuIsNotSupported,

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
