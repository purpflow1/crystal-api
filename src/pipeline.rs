use std::{marker::PhantomData, sync::Arc};

use crate::{
    GpuSamplerSet,
    mesh::{AttributeDescriptor, MeshBuffer},
    object::Object,
    proxies::PipelineProxy,
};

/// Empty struct used for compute pipeline as a template
pub struct ComputeDescriptor {}

impl AttributeDescriptor for ComputeDescriptor {
    fn get_attributes() -> &'static [crate::mesh::Attribute] {
        &[]
    }
}

/// Used to setup render or compute pipeline
pub struct Pipeline<V: AttributeDescriptor> {
    pub(crate) inner: Arc<dyn PipelineProxy>,
    _ty: PhantomData<V>,
}

impl<V: AttributeDescriptor> Pipeline<V> {
    pub(crate) fn new(proxy: Arc<dyn PipelineProxy>) -> Self {
        Self {
            inner: proxy,
            _ty: PhantomData,
        }
    }

    /// Creates graphics object with mesh only
    pub fn create_object_with_mesh<I>(&self, index: u32, mesh: &MeshBuffer<V, I>) -> Object {
        Object {
            pipeline: self.inner.clone(),
            mesh_buffer: Some(mesh.inner.clone()),
            sampler: None,
            groups: None,
            index,
            array: 1,
        }
    }

    /// Creates graphics object with mesh and textures
    pub fn create_object_with_mesh_sampled<I>(
        &self,
        index: u32,
        mesh: &MeshBuffer<V, I>,
        sampler: Arc<GpuSamplerSet>,
    ) -> Object {
        Object {
            pipeline: self.inner.clone(),
            mesh_buffer: Some(mesh.inner.clone()),
            sampler: Some(sampler),
            groups: None,
            index,
            array: 1,
        }
    }

    /// Creates array of graphics objects with mesh
    pub fn create_object_with_mesh_array<I>(
        &self,
        index: u32,
        mesh: &MeshBuffer<V, I>,
        array: u32,
    ) -> Object {
        Object {
            pipeline: self.inner.clone(),
            mesh_buffer: Some(mesh.inner.clone()),
            sampler: None,
            groups: None,
            index,
            array,
        }
    }

    /// Creates array of graphics objects with mesh and textures
    pub fn create_object_with_mesh_sampled_array<I>(
        &self,
        index: u32,
        mesh: &MeshBuffer<V, I>,
        sampler: Arc<GpuSamplerSet>,
        array: u32,
    ) -> Object {
        Object {
            pipeline: self.inner.clone(),
            mesh_buffer: Some(mesh.inner.clone()),
            sampler: Some(sampler),
            groups: None,
            index,
            array,
        }
    }
}

impl Pipeline<ComputeDescriptor> {
    /// Creates compute object. Need to specify compute groups
    pub fn create_object_compute(&self, groups: [u32; 3]) -> Object {
        Object {
            pipeline: self.inner.clone(),
            mesh_buffer: None,
            sampler: None,
            groups: Some(groups),
            index: 0,
            array: 0,
        }
    }

    /// Creates compute object with samplers
    pub fn create_object_compute_with_samplers(
        &self,
        groups: [u32; 3],
        sampler: Arc<GpuSamplerSet>,
    ) -> Object {
        Object {
            pipeline: self.inner.clone(),
            mesh_buffer: None,
            sampler: Some(sampler),
            groups: Some(groups),
            index: 0,
            array: 0,
        }
    }
}
