#[derive(Default)]
pub struct GraphicsApiInitSettings {
    pub msaa_samples: u8,
    pub width: u32,
    pub height: u32,
    pub bypass_capability_check: bool,
}

impl GraphicsApiInitSettings {
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

    pub fn bypass_capability_check(&self, bypass_capability_check: bool) -> Self {
        Self {
            bypass_capability_check,
            ..*self
        }
    }
}

pub struct DebugData {
    pub used_memory: u64,
    pub aviable_memory: u64,
}
