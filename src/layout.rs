use std::sync::Arc;

use crate::{
    GpuSamplerSet, Shader, buffer::Buffer, errors::GraphicsResult, mesh::Attribute,
    pipeline::Pipeline, proxies::LayoutProxy, render_target::RenderTarget,
};

pub struct Layout {
    pub(crate) inner: Arc<dyn LayoutProxy>,
}

impl Layout {
    pub(crate) fn new(proxy: Arc<dyn LayoutProxy>) -> Self {
        Self { inner: proxy }
    }

    /// Creates graphics pipeline
    pub fn create_graphics_pipeline(
        &self,
        render_target: &RenderTarget,
        shaders: &[Shader],
        attributes: &[Attribute],
    ) -> GraphicsResult<Pipeline> {
        Ok(Pipeline::new(self.inner.clone().create_graphics_pipeline(
            render_target.inner.clone(),
            shaders,
            attributes,
        )?))
    }

    /// Creates compute pipeline
    pub fn create_compute_pipeline(&self, shader: &Shader) -> GraphicsResult<Pipeline> {
        Ok(Pipeline::new(
            self.inner.clone().create_compute_pipeline(shader)?,
        ))
    }

    /// Registers samplers in layout for reusing them.
    /// Samplers are removing automatically on zero hard references in ```Arc```
    pub fn register_samplers(&self, samplers: &[Arc<GpuSamplerSet>]) -> GraphicsResult<()> {
        self.inner.register_samplers(samplers)
    }

    /// Adds buffer
    pub fn add_buffer<T>(&self, binding: u32, buffer: &Buffer<T>) -> GraphicsResult<()> {
        self.inner.add_buffer(binding, buffer.inner.clone())
    }
}
