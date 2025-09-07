use std::fmt::{Debug, Display};

/// ## Unified error enum
#[derive(Debug)]
pub enum GraphicsError {
    /// Happens on unexpected library error
    ConnotInitLibrary,
    /// Happens on software missing support
    NotSupportedSystem,
    /// Happens on hardware missing support
    NotSupportedDevice,
    /// Happens on hardware missing support or window server missing capability
    NotSupportedPresent,

    /// Happens on image present error
    PresentError,
    /// Happens on unified transfer/compute error
    TransferError,
    /// Happens on rendering error
    RenderingError,

    /// Happens on GPU sync error
    SyncError,
    /// Happens on resource error, for example on drop
    ResourceError,
    /// Happens on shader compilation/capability error
    ShaderError,
    /// Happens on GPU memory error
    MemoryError,
    /// Happens on unified shader data error
    DataError,
    /// Happens on image error
    ImageError,

    /// Happens on debug error
    DebugError,
}

impl Display for GraphicsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("{self:?}"))
    }
}

/// Type contains ```Result``` enum with ```GraphicsError``` on ```Err```
pub type GraphicsResult<T> = Result<T, GraphicsError>;
