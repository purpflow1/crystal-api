#[derive(Default)]
pub struct GraphicsApiInitSettings {
    pub viewport_frames_in_flight: u32,
    pub msaa_samples: u8,
    pub max_fps: u16,
    pub width: u32,
    pub height: u32,
}

impl GraphicsApiInitSettings {
    pub fn viewport_frames_in_flight(&self, viewport_frames_in_flight: u32) -> Self {
        Self {
            viewport_frames_in_flight,
            ..*self
        }
    }

    pub fn msaa_samples(&self, msaa_samples: u8) -> Self {
        Self {
            msaa_samples,
            ..*self
        }
    }

    pub fn max_fps(&self, max_fps: u16) -> Self {
        Self { max_fps, ..*self }
    }

    pub fn width(&self, width: u32) -> Self {
        Self { width, ..*self }
    }

    pub fn height(&self, height: u32) -> Self {
        Self { height, ..*self }
    }
}
