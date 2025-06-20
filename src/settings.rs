#[derive(Default)]
pub struct GraphicsApiInitSettings {
    pub double_buffering: bool,
    pub vsync: bool,
    pub msaa_samples: u8,
    pub width: u32,
    pub height: u32,
}

impl GraphicsApiInitSettings {
    pub fn double_buffering(&self, double_buffering: bool) -> Self {
        Self {
            double_buffering,
            ..*self
        }
    }

    pub fn vsync(&self, vsync: bool) -> Self {
        Self { vsync, ..*self }
    }

    pub fn msaa_samples(&self, msaa_samples: u8) -> Self {
        Self {
            msaa_samples,
            ..*self
        }
    }

    pub fn width(&self, width: u32) -> Self {
        Self { width, ..*self }
    }

    pub fn height(&self, height: u32) -> Self {
        Self { height, ..*self }
    }
}
