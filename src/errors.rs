use std::fmt::{Debug, Display};

#[derive(Debug)]
pub enum GraphicsError {
    ConnotInitLibrary,
    NotSupportedSystem,
    NotSupportedDevice,
    NotSupportedPresent,

    PresentError,
    TransferError,
    RenderingError,

    SyncError,
    ShaderError,
    MemoryError,
    DataError,
    ImageError,

    DebugError,

    OutOfDate,
}

impl Display for GraphicsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("{:?}", self))
    }
}

pub type GraphicsResult<T> = Result<T, GraphicsError>;
