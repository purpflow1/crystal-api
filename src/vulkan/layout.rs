use std::{
    cell::RefCell,
    collections::{BTreeMap, VecDeque},
    sync::Arc,
};

use vulkano::{
    buffer::Subbuffer,
    command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer},
    descriptor_set::{
        DescriptorSet, WriteDescriptorSet,
        allocator::{StandardDescriptorSetAllocator, StandardDescriptorSetAllocatorCreateInfo},
        layout::{
            DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo,
            DescriptorType,
        },
    },
    device::{Device, DeviceOwned},
    image::sampler::{
        BorderColor, Filter, Sampler, SamplerAddressMode, SamplerCreateInfo, SamplerMipmapMode,
    },
    memory::allocator::StandardMemoryAllocator,
    pipeline::{PipelineBindPoint, PipelineLayout, layout::PipelineLayoutCreateInfo},
    shader::ShaderStages,
};

use crate::{
    debug::log,
    errors::{CrystalError, CrystalResult},
    object::Object,
    traits,
    vulkan::{VulkanTexture, images::Image},
};

use super::{memory_obj::VulkanObjectMemoryManager, rendering::VulkanRenderTarget};

pub struct VulkanLayout {
    allocator: Arc<StandardDescriptorSetAllocator>,

    pub pipeline_layout: Arc<PipelineLayout>,
    descriptors: BTreeMap<(i32, usize), Arc<DescriptorSet>>,
    samplers: BTreeMap<u32, Arc<Sampler>>,

    object_render_queue: VecDeque<Arc<RefCell<Object>>>,
}

impl traits::Layout for VulkanLayout {
    fn as_vulkan_mut(&mut self) -> Option<&mut super::VulkanLayout> {
        Some(self)
    }

    fn as_vulkan_ref(&self) -> Option<&super::VulkanLayout> {
        Some(self)
    }

    fn add_object_to_queue(&mut self, object: Arc<RefCell<Object>>) {
        self.object_render_queue.push_back(object);
    }
}

impl VulkanLayout {
    pub(crate) fn new(
        device: Arc<Device>,
        sampler_binding_num: usize,
        uniform_binding_num: usize,
        storage_binding_num: usize,
    ) -> CrystalResult<Self> {
        let uniform_bindings_enum = vec![
            DescriptorSetLayoutBinding {
                descriptor_count: uniform_binding_num as u32,
                ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::UniformBuffer)
            };
            uniform_binding_num
        ];
        let storage_bindings_enum = vec![
            DescriptorSetLayoutBinding {
                descriptor_count: storage_binding_num as u32,
                ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::StorageBuffer)
            };
            storage_binding_num
        ];
        let sampler_bindings_enum = vec![
            DescriptorSetLayoutBinding {
                descriptor_count: sampler_binding_num as u32,
                stages: ShaderStages::FRAGMENT,
                ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::CombinedImageSampler)
            };
            sampler_binding_num
        ];

        let mut sampler_bindings = BTreeMap::new();
        let mut uniform_bindings = BTreeMap::new();
        let mut storage_bindings = BTreeMap::new();

        for (idx, sampler_binding) in sampler_bindings_enum.iter().enumerate() {
            sampler_bindings.insert(idx as u32, sampler_binding.clone());
        }

        for (idx, uniform_binding) in uniform_bindings_enum.iter().enumerate() {
            uniform_bindings.insert(idx as u32, uniform_binding.clone());
        }

        for (idx, storage_binding) in storage_bindings_enum.iter().enumerate() {
            storage_bindings.insert(idx as u32, storage_binding.clone());
        }

        let uniform_descriptor_set_layout_create_info = DescriptorSetLayoutCreateInfo {
            bindings: uniform_bindings,
            ..Default::default()
        };

        let storage_descriptor_set_layout_create_info = DescriptorSetLayoutCreateInfo {
            bindings: storage_bindings,
            ..Default::default()
        };

        let sampler_descriptor_set_layout_create_info = DescriptorSetLayoutCreateInfo {
            bindings: sampler_bindings,
            ..Default::default()
        };

        let uniform_descriptor_set_layout = match DescriptorSetLayout::new(
            device.clone(),
            uniform_descriptor_set_layout_create_info,
        ) {
            Ok(descriptor_set_layout) => descriptor_set_layout,
            Err(e) => {
                log!("cannot create descriptor set layout: {:?}", e);
                return Err(CrystalError::DescriptorError);
            }
        };

        let storage_descriptor_set_layout = match DescriptorSetLayout::new(
            device.clone(),
            storage_descriptor_set_layout_create_info,
        ) {
            Ok(descriptor_set_layout) => descriptor_set_layout,
            Err(e) => {
                log!("cannot create descriptor set layout: {:?}", e);
                return Err(CrystalError::DescriptorError);
            }
        };

        let sampler_descriptor_set_layout = match DescriptorSetLayout::new(
            device.clone(),
            sampler_descriptor_set_layout_create_info,
        ) {
            Ok(descriptor_set_layout) => descriptor_set_layout,
            Err(e) => {
                log!("cannot create descriptor set layout: {:?}", e);
                return Err(CrystalError::DescriptorError);
            }
        };

        let create_info = PipelineLayoutCreateInfo {
            set_layouts: vec![
                uniform_descriptor_set_layout,
                storage_descriptor_set_layout,
                sampler_descriptor_set_layout,
            ],
            ..Default::default()
        };

        let pipeline_layout = match PipelineLayout::new(device.clone(), create_info) {
            Ok(layout) => layout,
            Err(e) => {
                log!("cannot create pipeline layout: {:?}", e);
                return Err(CrystalError::DescriptorError);
            }
        };

        let create_info = StandardDescriptorSetAllocatorCreateInfo::default();
        let allocator = Arc::new(StandardDescriptorSetAllocator::new(device, create_info));

        Ok(Self {
            allocator,

            pipeline_layout,
            samplers: BTreeMap::new(),
            descriptors: BTreeMap::new(),

            object_render_queue: VecDeque::new(),
        })
    }

    #[deprecated]
    #[allow(unreachable_code, unused_variables)]
    pub(crate) fn new_static(
        memory_allocator: Arc<StandardMemoryAllocator>,
        image_views_sampled_num: u32,
        frames_in_flight: u32,
        max_instance_num: u64,

        buffers: &[(bool, u64)],
    ) -> CrystalResult<Self> {
        unimplemented!();
    }

    fn update_descriptor(
        &mut self,
        cur_frame: usize,
        descriptor_type: DescriptorType,
        descriptor_write: WriteDescriptorSet,
    ) -> CrystalResult<()> {
        match self
            .descriptors
            .iter()
            .find(|((typ, frame), _)| *typ == descriptor_type as i32 && *frame == cur_frame)
        {
            Some((_, descriptor)) => unsafe {
                descriptor.update_by_ref([descriptor_write], []).unwrap()
            },
            None => {
                let layout = self
                    .pipeline_layout
                    .set_layouts()
                    .iter()
                    .find(|layout| {
                        layout
                            .bindings()
                            .first_key_value()
                            .unwrap()
                            .1
                            .descriptor_type
                            == descriptor_type
                    })
                    .unwrap();

                let descriptor = DescriptorSet::new(
                    self.allocator.clone(),
                    layout.clone(),
                    [descriptor_write],
                    [],
                )
                .unwrap();

                self.descriptors
                    .insert((descriptor_type as i32, cur_frame), descriptor);
            }
        };

        Ok(())
    }

    pub(crate) fn update_descriptor_buffer(
        &mut self,
        frame: usize,
        descriptor_type: DescriptorType,
        binding: u32,
        buffer: Subbuffer<impl ?Sized>,
    ) -> CrystalResult<()> {
        let descriptor_write = WriteDescriptorSet::buffer(binding, buffer);

        self.update_descriptor(frame, descriptor_type, descriptor_write)
    }

    pub(crate) fn update_descriptor_texture(
        &mut self,
        frame: usize,
        binding: u32,
        texture: Arc<VulkanTexture>,
    ) -> CrystalResult<()> {
        let sampler = self.get_sampler(&texture.image)?;

        let descriptor_write = WriteDescriptorSet::image_view_sampler(
            binding,
            texture.image.image_view.clone(),
            sampler,
        );

        self.update_descriptor(
            frame,
            DescriptorType::CombinedImageSampler,
            descriptor_write,
        )
    }

    pub(crate) fn render(
        &mut self,
        command_buffer: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        render_target: &VulkanRenderTarget,
    ) -> CrystalResult<()> {
        let frame_descriptor_sets: Vec<(i32, Arc<DescriptorSet>)> = self
            .descriptors
            .iter()
            .filter(|((_, frame), _)| *frame == render_target.current_frame)
            .map(|((typ, _), val)| (*typ, val.clone()))
            .collect();

        let uniform_descriptor_set: Arc<DescriptorSet> = frame_descriptor_sets
            .iter()
            .find(|(desc_type, _)| *desc_type == DescriptorType::UniformBuffer as i32)
            .unwrap()
            .1
            .clone();

        let storage_descriptor_set: Arc<DescriptorSet> = frame_descriptor_sets
            .iter()
            .find(|(desc_type, _)| *desc_type == DescriptorType::StorageBuffer as i32)
            .unwrap()
            .1
            .clone();

        match command_buffer.bind_descriptor_sets(
            PipelineBindPoint::Graphics,
            self.pipeline_layout.clone(),
            0,
            vec![uniform_descriptor_set, storage_descriptor_set],
        ) {
            Ok(_) => (),
            Err(e) => {
                log!("cannot bind descriptor sets: {:?}", e);
                return Err(CrystalError::RenderingError);
            }
        };

        let mut current_object_idx = 0usize;

        while self.object_render_queue.len() > 0 {
            let obj = self.object_render_queue.pop_front().unwrap();
            let mut obj = obj.borrow_mut();

            match obj.mesh.clone() {
                Some(mesh) => match obj.memory_manager {
                    Some(_) => {}
                    None => {
                        obj.memory_manager = Some(
                            VulkanObjectMemoryManager::new(
                                self.allocator.device().clone(),
                                &mesh.vertices,
                                &mesh.indices,
                            )
                            .unwrap(),
                        );
                    }
                },
                None => {}
            }

            match &obj.textures {
                None => {}
                Some(textures) => {
                    for (binding, texture) in textures {
                        self.update_descriptor_texture(
                            render_target.current_frame,
                            *binding,
                            texture.clone().as_vulkan_arc().unwrap(),
                        )?;
                    }

                    let sampler_descriptor_set: Arc<DescriptorSet> = frame_descriptor_sets
                        .iter()
                        .find(|(desc_type, _)| {
                            *desc_type == DescriptorType::CombinedImageSampler as i32
                        })
                        .unwrap()
                        .1
                        .clone();

                    match command_buffer.bind_descriptor_sets(
                        PipelineBindPoint::Graphics,
                        self.pipeline_layout.clone(),
                        2,
                        vec![sampler_descriptor_set],
                    ) {
                        Ok(_) => (),
                        Err(e) => {
                            log!("cannot bind descriptor sets: {:?}", e);
                            return Err(CrystalError::RenderingError);
                        }
                    };
                }
            }

            let object_memory_manager = match obj.memory_manager.as_ref().unwrap().as_vulkan_ref() {
                Some(mm) => mm,
                None => panic!("fatal: wrong object memory manager type, expected vulkan"),
            };

            let pipeline = match obj.pipeline.clone().as_vulkan() {
                Some(pipeline) => pipeline,
                None => panic!("fatal: wrong pipeline type, expected vulkan"),
            };

            let index_buffer = object_memory_manager.index_buffer_manager.buffer.clone();
            let vertex_buffer = object_memory_manager.vertex_buffer_manager.buffer.clone();

            match command_buffer.bind_pipeline_graphics(pipeline) {
                Ok(_) => (),
                Err(e) => {
                    log!("cannot bind graphics pipeline: {:?}", e);
                    return Err(CrystalError::RenderingError);
                }
            };

            match command_buffer.bind_index_buffer((*index_buffer).clone()) {
                Ok(_) => (),
                Err(e) => {
                    log!("cannot bind index buffer: {:?}", e);
                    return Err(CrystalError::RenderingError);
                }
            };

            match command_buffer.bind_vertex_buffers(0, vec![(*vertex_buffer).clone()]) {
                Ok(_) => (),
                Err(e) => {
                    log!("cannot bind vertex buffer: {:?}", e);
                    return Err(CrystalError::RenderingError);
                }
            };

            let index_count = obj.mesh.as_ref().unwrap().indices.len() as u32;

            match unsafe {
                command_buffer.draw_indexed(index_count, 1, 0, 0, current_object_idx as u32)
            } {
                Ok(_) => (),
                Err(e) => {
                    log!("cannot draw indexed: {:?}", e);
                    return Err(CrystalError::RenderingError);
                }
            };

            current_object_idx += 1;
        }

        Ok(())
    }

    fn get_sampler(&mut self, image: &Image) -> CrystalResult<Arc<Sampler>> {
        let mip_levels = image.mip_levels;
        let anisotropy_texels = image.anisotropy_texels;

        match self.samplers.get(&mip_levels) {
            Some(sampler) => return Ok(sampler.clone()),
            None => (),
        };

        let create_info = SamplerCreateInfo {
            mag_filter: Filter::Linear,
            min_filter: Filter::Linear,
            address_mode: [SamplerAddressMode::Repeat; 3],
            anisotropy: Some(anisotropy_texels),
            border_color: BorderColor::IntOpaqueBlack,
            unnormalized_coordinates: false,
            compare: None,
            mipmap_mode: SamplerMipmapMode::Linear,
            mip_lod_bias: 0.,
            lod: 0.0..=mip_levels as f32,
            ..Default::default()
        };

        let sampler = match Sampler::new(self.allocator.device().clone(), create_info) {
            Ok(sampler) => sampler,
            Err(e) => {
                log!("cannot create sampler: {:?}", e);
                return Err(CrystalError::ImageError);
            }
        };

        self.samplers.insert(mip_levels, sampler.clone());

        Ok(sampler)
    }
}
