use std::sync::Arc;

use crate::proxies::PipelineProxy;

pub struct Pipeline {
    pub(crate) inner: Arc<dyn PipelineProxy>,
}

impl Pipeline {
    pub(crate) fn new(proxy: Arc<dyn PipelineProxy>) -> Self {
        Self { inner: proxy }
    }
}
