use std::sync::Arc;

use crate::proxies::TextureProxy;

pub struct Texture {
    pub(crate) inner: Arc<dyn TextureProxy>,
}

impl Texture {
    pub(crate) fn new(proxy: Arc<dyn TextureProxy>) -> Self {
        Self { inner: proxy }
    }
}
