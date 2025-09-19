use std::sync::Arc;

use crate::{debug::log, errors::GraphicsResult, proxies::RenderTargetProxy, texture::Texture};

/// Used to render into
pub struct RenderTarget {
    pub(crate) inner: Arc<dyn RenderTargetProxy>,
}

impl RenderTarget {
    pub(crate) fn new(proxy: Arc<dyn RenderTargetProxy>) -> Self {
        Self { inner: proxy }
    }

    /// Inherits a new `RenderTarget` from current
    pub fn inherit(
        &self,
        extent: [u32; 2],
        anisotropy_texels: f32,
        msaa_samples: u8,
    ) -> GraphicsResult<(Self, Texture)> {
        log!(
            "creating render target [ extent: {}x{} ]",
            extent[0],
            extent[1]
        );

        let (render_target, texture) =
            self.inner
                .create_render_target(extent, anisotropy_texels, msaa_samples)?;

        Ok((
            Self {
                inner: render_target,
            },
            Texture::new(texture),
        ))
    }
}
