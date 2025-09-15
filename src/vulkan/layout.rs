use std::{
    collections::BTreeMap,
    ffi::CString,
    iter::zip,
    sync::{Arc, Mutex},
};

use ash::vk;

use crate::{
    GpuSamplerSet, Shader, ShaderStage,
    debug::{error, log},
    errors::{GraphicsError, GraphicsResult},
    mesh::Attribute,
    object::Object,
    proxies::*,
};

use super::{VulkanRenderTarget, devices::DeviceManager};

struct LayoutDynamicData {
    device_manager: Arc<DeviceManager>,

    uniform_descriptor_sets: Vec<vk::DescriptorSet>,
    storage_descriptor_sets: Vec<vk::DescriptorSet>,
    sampler_descriptor_sets: Vec<vk::DescriptorSet>,

    n_pass: usize,
    double_buffering: bool,

    sampler_binding_data: BTreeMap<usize, Arc<GpuSamplerSet>>,
    samplers: BTreeMap<u32, vk::Sampler>,
}

unsafe impl Sync for LayoutDynamicData {}
unsafe impl Send for LayoutDynamicData {}

impl Drop for LayoutDynamicData {
    fn drop(&mut self) {
        unsafe {
            self.samplers.iter().for_each(|(_, &sampler)| {
                self.device_manager.device.destroy_sampler(sampler, None)
            });
        }
    }
}

impl LayoutDynamicData {
    fn update_data(&mut self) -> GraphicsResult<usize> {
        let indices_to_clean: Vec<usize> = self
            .sampler_binding_data
            .iter()
            .filter(|(_, sampler)| Arc::strong_count(sampler) == 1)
            .map(|(ind, _)| *ind)
            .collect();

        for ind in &indices_to_clean {
            self.sampler_binding_data.remove(ind);
        }

        let result = indices_to_clean.len();

        if self.double_buffering {
            self.n_pass = (self.n_pass + 1) % 2;
        }

        Ok(result)
    }

    fn add_textures(&mut self, sampler_set: Arc<GpuSamplerSet>) -> GraphicsResult<()> {
        let mut id_lock = sampler_set.id.lock().unwrap();
        if *id_lock != usize::MAX {
            match self.sampler_binding_data.remove(&*id_lock) {
                Some(_sampler) => (),
                None => {
                    error!("sampler is already bound to another layout");
                    return Err(GraphicsError::TransferError);
                }
            }
        }

        *id_lock = (0..usize::MAX)
            .find(|idx| !self.sampler_binding_data.iter().any(|(x, _)| *x == *idx))
            .unwrap();

        self.sampler_binding_data
            .insert(*id_lock, sampler_set.clone());

        for (binding, texture) in &sampler_set.textures {
            let vulkan_texture = texture.clone().as_vulkan().unwrap();

            let sampler = match self.samplers.get(&vulkan_texture.image.mip_levels) {
                Some(sampler) => *sampler,
                None => {
                    let sampler_info = vk::SamplerCreateInfo::default()
                        .mag_filter(vk::Filter::LINEAR)
                        .min_filter(vk::Filter::LINEAR)
                        .address_mode_u(vk::SamplerAddressMode::REPEAT)
                        .address_mode_v(vk::SamplerAddressMode::REPEAT)
                        .address_mode_w(vk::SamplerAddressMode::REPEAT)
                        .anisotropy_enable(true)
                        .anisotropy_enable(vulkan_texture.image.anisotropy_texels > 1.)
                        .max_anisotropy(vulkan_texture.image.anisotropy_texels)
                        .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
                        .unnormalized_coordinates(false)
                        .compare_enable(false)
                        .compare_op(vk::CompareOp::ALWAYS)
                        .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                        .mip_lod_bias(0.)
                        .min_lod(0.)
                        .max_lod(vulkan_texture.image.mip_levels as f32);

                    match unsafe {
                        self.device_manager
                            .device
                            .create_sampler(&sampler_info, None)
                    } {
                        Ok(sampler) => {
                            self.samplers
                                .insert(vulkan_texture.image.mip_levels, sampler);
                            sampler
                        }
                        Err(e) => {
                            error!("cannot create sampler: {}", e);
                            return Err(GraphicsError::ImageError);
                        }
                    }
                }
            };

            let image_infos = [vk::DescriptorImageInfo::default()
                .image_view(vulkan_texture.image.image_view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .sampler(sampler)];

            let descriptor_write = {
                let descriptor_set = self.sampler_descriptor_sets[*id_lock];

                vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(*binding)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&image_infos)
            };

            unsafe {
                self.device_manager
                    .device
                    .update_descriptor_sets(&[descriptor_write], &[])
            };
        }

        Ok(())
    }

    fn add_buffer(&mut self, binding: u32, buffer: Arc<dyn BufferProxy>) -> GraphicsResult<()> {
        let buffer = buffer.as_vulkan().clone().unwrap();

        let size = buffer.info.size;
        let buffer_usage = buffer.info.usage;

        let descriptor_type = if buffer_usage.contains(vk::BufferUsageFlags::UNIFORM_BUFFER) {
            vk::DescriptorType::UNIFORM_BUFFER
        } else {
            vk::DescriptorType::STORAGE_BUFFER
        };

        for (idx, buffer_handler) in buffer.get_handlers().iter().enumerate() {
            let buffer_info = vk::DescriptorBufferInfo::default()
                .buffer(*buffer_handler)
                .offset(0)
                .range(size);

            let buffer_infos = &[buffer_info];

            let descriptor_set = if buffer_usage.contains(vk::BufferUsageFlags::UNIFORM_BUFFER) {
                self.uniform_descriptor_sets[idx]
            } else {
                self.storage_descriptor_sets[idx]
            };

            let descriptor_write = vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(binding)
                .dst_array_element(0)
                .descriptor_type(descriptor_type)
                .descriptor_count(1)
                .buffer_info(buffer_infos);

            unsafe {
                self.device_manager
                    .device
                    .update_descriptor_sets(&[descriptor_write], &[])
            };
        }

        if buffer.info.count == 1 && self.double_buffering {
            let descriptor_set = if buffer_usage.contains(vk::BufferUsageFlags::UNIFORM_BUFFER) {
                self.uniform_descriptor_sets[1]
            } else {
                self.storage_descriptor_sets[1]
            };

            let buffer_info = vk::DescriptorBufferInfo::default()
                .buffer(buffer.get_handlers()[0])
                .offset(0)
                .range(size);

            let buffer_infos = &[buffer_info];

            let descriptor_write = vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(binding)
                .dst_array_element(0)
                .descriptor_type(descriptor_type)
                .descriptor_count(1)
                .buffer_info(buffer_infos);

            unsafe {
                self.device_manager
                    .device
                    .update_descriptor_sets(&[descriptor_write], &[])
            };
        }

        Ok(())
    }
}

pub struct VulkanLayout {
    device_manager: Arc<DeviceManager>,

    descriptor_pool: vk::DescriptorPool,
    descriptor_set_layouts: Vec<vk::DescriptorSetLayout>,

    pub pipeline_layout: vk::PipelineLayout,

    dynamic_data: Mutex<LayoutDynamicData>,
}

impl Drop for VulkanLayout {
    fn drop(&mut self) {
        unsafe {
            self.descriptor_set_layouts.iter().for_each(|&layout| {
                self.device_manager
                    .device
                    .destroy_descriptor_set_layout(layout, None)
            });

            self.device_manager
                .device
                .destroy_descriptor_pool(self.descriptor_pool, None);

            self.device_manager
                .device
                .destroy_pipeline_layout(self.pipeline_layout, None);
        }
    }
}

impl LayoutProxy for VulkanLayout {
    fn add_buffer(&self, binding: u32, buffer: Arc<dyn BufferProxy>) -> GraphicsResult<()> {
        let mut lock = self.dynamic_data.lock().unwrap();
        lock.add_buffer(binding, buffer)
    }

    fn register_samplers(&self, samplers: &[Arc<GpuSamplerSet>]) -> GraphicsResult<()> {
        let mut dynamic_data = self.dynamic_data.lock().unwrap();

        samplers
            .iter()
            .for_each(|sampler| dynamic_data.add_textures(sampler.clone()).unwrap());
        Ok(())
    }

    fn create_graphics_pipeline(
        self: Arc<Self>,
        render_target: Arc<dyn RenderTargetProxy>,
        shaders: &[Shader],
        attributes: &[Attribute],
    ) -> GraphicsResult<Arc<dyn PipelineProxy>> {
        Ok(VulkanPipeline::from_render_target(
            self.device_manager.clone(),
            self,
            shaders,
            attributes,
            render_target.as_vulkan().unwrap(),
        )?)
    }

    fn create_compute_pipeline(
        self: Arc<Self>,
        shader: &Shader,
    ) -> GraphicsResult<Arc<dyn PipelineProxy>> {
        Ok(VulkanPipeline::new_compute(
            self.device_manager.clone(),
            self,
            shader,
        )?)
    }
}

impl VulkanLayout {
    pub(crate) fn update_dynamic_data(&self) -> GraphicsResult<usize> {
        self.dynamic_data.lock().unwrap().update_data()
    }

    pub(crate) fn new(
        device_manager: Arc<DeviceManager>,
        texture_num: usize,
        sampler_num: usize,
        uniform_num: usize,
        storage_num: usize,

        double_buffering: bool,
    ) -> GraphicsResult<Arc<Self>> {
        let buffer_count = if double_buffering { 2 } else { 1 };

        let mut pool_sizes = vec![];

        if uniform_num > 0 {
            pool_sizes.push(
                vk::DescriptorPoolSize::default()
                    .descriptor_count(buffer_count * uniform_num as u32)
                    .ty(vk::DescriptorType::UNIFORM_BUFFER),
            )
        }

        if storage_num > 0 {
            pool_sizes.push(
                vk::DescriptorPoolSize::default()
                    .descriptor_count(buffer_count * storage_num as u32)
                    .ty(vk::DescriptorType::STORAGE_BUFFER),
            )
        }

        if sampler_num + texture_num > 0 {
            pool_sizes.push(
                vk::DescriptorPoolSize::default()
                    .descriptor_count((sampler_num * texture_num) as u32)
                    .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER),
            )
        }

        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(buffer_count * (uniform_num + storage_num + sampler_num) as u32)
            .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND);

        let descriptor_pool = match unsafe {
            device_manager
                .device
                .create_descriptor_pool(&pool_info, None)
        } {
            Ok(descriptor_pool) => descriptor_pool,
            Err(e) => {
                error!("cannot create descriptor pool: {}", e);
                return Err(GraphicsError::DataError);
            }
        };

        let ubo_bindings: Vec<_> = (0..uniform_num)
            .map(|idx| {
                vk::DescriptorSetLayoutBinding::default()
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .stage_flags(vk::ShaderStageFlags::ALL)
                    .descriptor_count(1)
                    .binding(idx as u32)
            })
            .collect();

        let ssbo_bindings: Vec<_> = (0..storage_num)
            .map(|idx| {
                vk::DescriptorSetLayoutBinding::default()
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .stage_flags(vk::ShaderStageFlags::ALL)
                    .descriptor_count(1)
                    .binding(idx as u32)
            })
            .collect();

        let sampler_bindings: Vec<_> = (0..texture_num)
            .map(|idx| {
                vk::DescriptorSetLayoutBinding::default()
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                    .descriptor_count(1)
                    .binding(idx as u32)
            })
            .collect();

        let ubo_descriptor_set_layout_create_info =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(&ubo_bindings);

        let ssbo_descriptor_set_layout_create_info =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(&ssbo_bindings);

        let sampler_descriptor_set_layout_create_info =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(&sampler_bindings);

        let ubo_descriptor_set_layout = match unsafe {
            device_manager
                .device
                .clone()
                .create_descriptor_set_layout(&ubo_descriptor_set_layout_create_info, None)
        } {
            Ok(descriptor_set_layout) => descriptor_set_layout,
            Err(e) => {
                error!("cannot create descriptor set layout: {}", e);
                return Err(GraphicsError::DataError);
            }
        };

        let ssbo_descriptor_set_layout = match unsafe {
            device_manager
                .device
                .clone()
                .create_descriptor_set_layout(&ssbo_descriptor_set_layout_create_info, None)
        } {
            Ok(descriptor_set_layout) => descriptor_set_layout,
            Err(e) => {
                error!("cannot create descriptor set layout: {}", e);
                return Err(GraphicsError::DataError);
            }
        };

        let sampler_descriptor_set_layout = match unsafe {
            device_manager
                .device
                .clone()
                .create_descriptor_set_layout(&sampler_descriptor_set_layout_create_info, None)
        } {
            Ok(descriptor_set_layout) => descriptor_set_layout,
            Err(e) => {
                error!("cannot create descriptor set layout: {}", e);
                return Err(GraphicsError::DataError);
            }
        };

        let ubo_layouts = vec![ubo_descriptor_set_layout; buffer_count as usize];
        let ssbo_layouts = vec![ssbo_descriptor_set_layout; buffer_count as usize];
        let sampler_layouts = vec![sampler_descriptor_set_layout; sampler_num];

        let ubo_alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&ubo_layouts);

        let ssbo_alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&ssbo_layouts);

        let sampler_alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&sampler_layouts);

        let uniform_descriptor_sets = if uniform_num > 0 {
            match unsafe {
                device_manager
                    .device
                    .allocate_descriptor_sets(&ubo_alloc_info)
            } {
                Ok(descriptor_sets) => descriptor_sets,
                Err(e) => {
                    error!("cannot allocate uniform descriptor sets: {}", e);
                    return Err(GraphicsError::DataError);
                }
            }
        } else {
            vec![]
        };

        let storage_descriptor_sets = if storage_num > 0 {
            match unsafe {
                device_manager
                    .device
                    .allocate_descriptor_sets(&ssbo_alloc_info)
            } {
                Ok(descriptor_sets) => descriptor_sets,
                Err(e) => {
                    error!("cannot allocate storage descriptor sets: {}", e);
                    return Err(GraphicsError::DataError);
                }
            }
        } else {
            vec![]
        };

        let sampler_descriptor_sets = if sampler_num > 0 {
            match unsafe {
                device_manager
                    .device
                    .allocate_descriptor_sets(&sampler_alloc_info)
            } {
                Ok(descriptor_sets) => descriptor_sets,
                Err(e) => {
                    error!("cannot allocate sampler descriptor sets: {}", e);
                    return Err(GraphicsError::DataError);
                }
            }
        } else {
            vec![]
        };

        let descriptor_set_layouts = vec![
            ubo_descriptor_set_layout,
            ssbo_descriptor_set_layout,
            sampler_descriptor_set_layout,
        ];

        let create_info =
            vk::PipelineLayoutCreateInfo::default().set_layouts(&descriptor_set_layouts);

        let pipeline_layout = match unsafe {
            device_manager
                .device
                .clone()
                .create_pipeline_layout(&create_info, None)
        } {
            Ok(layout) => layout,
            Err(e) => {
                error!("cannot create pipeline layout: {}", e);
                return Err(GraphicsError::DataError);
            }
        };

        Ok(Arc::new(Self {
            device_manager: device_manager.clone(),

            descriptor_pool,
            descriptor_set_layouts,

            pipeline_layout,

            dynamic_data: Mutex::new(LayoutDynamicData {
                device_manager,

                uniform_descriptor_sets,
                storage_descriptor_sets,
                sampler_descriptor_sets,

                n_pass: 0,
                double_buffering,

                sampler_binding_data: BTreeMap::new(),

                samplers: BTreeMap::new(),
            }),
        }))
    }

    pub(crate) fn get_descriptor_sets(&self) -> Vec<vk::DescriptorSet> {
        let dynamic_data = self.dynamic_data.lock().unwrap();
        vec![
            dynamic_data.uniform_descriptor_sets[dynamic_data.n_pass],
            dynamic_data.storage_descriptor_sets[dynamic_data.n_pass],
        ]
    }

    pub(crate) fn render(
        &self,
        objects: &[Arc<Object>],
        command_buffer: &vk::CommandBuffer,
    ) -> GraphicsResult<()> {
        let dynamic_data = self.dynamic_data.lock().unwrap();
        let device_manager = dynamic_data.device_manager.clone();

        unsafe {
            device_manager.device.cmd_bind_descriptor_sets(
                *command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &[
                    dynamic_data.uniform_descriptor_sets[dynamic_data.n_pass],
                    dynamic_data.storage_descriptor_sets[dynamic_data.n_pass],
                ],
                &[],
            )
        }

        let mut result = Ok(());

        for object in objects.iter() {
            if let Some(sampler) = &object.sampler {
                let id = *sampler.id.lock().unwrap();

                if id == usize::MAX {
                    error!("object sampler has not been registered!");
                    result = Err(GraphicsError::DataError)
                }

                unsafe {
                    device_manager.device.cmd_bind_descriptor_sets(
                        *command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.pipeline_layout,
                        2,
                        &[dynamic_data.sampler_descriptor_sets[id]],
                        &[],
                    )
                }
            }

            let pipeline = match object.pipeline.clone().as_vulkan() {
                Some(pipeline) => pipeline,
                None => {
                    error!("wrong pipeline type, expected vulkan");
                    return Err(GraphicsError::DataError);
                }
            };

            let mesh_buffer = object.mesh_buffer.as_ref().unwrap();

            let index = mesh_buffer.indices.clone().as_vulkan().unwrap();

            let index_buffer = index.get_handlers()[0];
            let vertex_buffer = mesh_buffer
                .vertices
                .clone()
                .as_vulkan()
                .unwrap()
                .get_handlers()[0];

            let index_type = match mesh_buffer.index_size {
                1 => vk::IndexType::UINT8_KHR,
                2 => vk::IndexType::UINT16,
                4 => vk::IndexType::UINT32,
                size => {
                    error!("unknown index size: {}", size);
                    vk::IndexType::NONE_KHR
                }
            };

            unsafe {
                device_manager.device.cmd_bind_pipeline(
                    *command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipeline.handle,
                );

                device_manager.device.cmd_bind_index_buffer(
                    *command_buffer,
                    index_buffer,
                    0,
                    vk::IndexType::UINT32,
                );
            }

            let index_count =
                index.as_vulkan().as_ref().unwrap().info.size / mesh_buffer.index_size as u64;

            unsafe {
                device_manager.device.cmd_bind_vertex_buffers(
                    *command_buffer,
                    0,
                    &[vertex_buffer],
                    &[0],
                )
            }

            let instance_count = object.array;

            unsafe {
                device_manager.device.cmd_draw_indexed(
                    *command_buffer,
                    index_count as u32,
                    instance_count,
                    0,
                    0,
                    object.index,
                )
            }
        }

        result
    }
}

#[derive(Clone)]
struct ShaderStageInfo {
    device_manager: Arc<DeviceManager>,
    module: vk::ShaderModule,
    stage: vk::ShaderStageFlags,
    entry_point: CString,
}

impl Drop for ShaderStageInfo {
    fn drop(&mut self) {
        unsafe {
            self.device_manager
                .device
                .destroy_shader_module(self.module, None);
        }
    }
}

impl ShaderStageInfo {
    pub(crate) fn as_vk<'a>(&self) -> vk::PipelineShaderStageCreateInfo<'a> {
        vk::PipelineShaderStageCreateInfo {
            stage: self.stage,
            module: self.module,
            p_name: self.entry_point.as_ptr(),
            ..Default::default()
        }
    }
}

pub struct VulkanPipeline {
    device_manager: Arc<DeviceManager>,
    pub(crate) layout: Arc<VulkanLayout>,
    pub(crate) render_target: Option<Arc<VulkanRenderTarget>>,
    pub handle: vk::Pipeline,
}

impl Drop for VulkanPipeline {
    fn drop(&mut self) {
        unsafe {
            self.device_manager
                .device
                .destroy_pipeline(self.handle, None);
        }
    }
}

impl VulkanPipeline {
    fn stages_as_vk<'a>(
        stages: impl IntoIterator<Item = Arc<ShaderStageInfo>>,
    ) -> Vec<vk::PipelineShaderStageCreateInfo<'a>> {
        stages.into_iter().map(|stage| stage.as_vk()).collect()
    }

    pub fn new_compute(
        device_manager: Arc<DeviceManager>,
        layout: Arc<VulkanLayout>,
        shader: &Shader,
    ) -> GraphicsResult<Arc<Self>> {
        log!("creating compute pipeline");

        let stage = match shader.stage {
            ShaderStage::Compute => vk::ShaderStageFlags::COMPUTE,
            _ => {
                log!(
                    "wrong shader stage specified in compute pipeline: {:?}",
                    shader.stage
                );
                return Err(GraphicsError::ShaderError);
            }
        };

        let shader_module_create_info = vk::ShaderModuleCreateInfo::default().code(&shader.code);

        let module = match unsafe {
            device_manager
                .device
                .create_shader_module(&shader_module_create_info, None)
        } {
            Ok(module) => module,
            Err(e) => {
                error!("cannot create shader module: {}", e);
                return Err(GraphicsError::ShaderError);
            }
        };

        let shader_stage_info = Arc::new(ShaderStageInfo {
            device_manager: device_manager.clone(),
            module,
            stage,
            entry_point: CString::new("main").unwrap(),
        });

        let create_info = vk::ComputePipelineCreateInfo::default()
            .layout(layout.pipeline_layout)
            .stage(shader_stage_info.as_vk());

        let pipeline = match unsafe {
            device_manager.device.create_compute_pipelines(
                vk::PipelineCache::null(),
                &[create_info],
                None,
            )
        } {
            Ok(pipelines) => pipelines[0],
            Err(e) => {
                error!("cannot create compute pipeline: {:?}", e);
                return Err(GraphicsError::ShaderError);
            }
        };

        Ok(Arc::new(Self {
            device_manager,
            layout,
            handle: pipeline,
            render_target: None,
        }))
    }

    pub fn from_render_target(
        device_manager: Arc<DeviceManager>,
        layout: Arc<VulkanLayout>,
        shaders: &[Shader],
        attributes: &[Attribute],
        render_target: Arc<VulkanRenderTarget>,
    ) -> GraphicsResult<Arc<Self>> {
        log!("creating graphics pipeline");

        if shaders.is_empty() {
            error!("no shaders specified");
            return Err(GraphicsError::ShaderError);
        }

        let mut stages = Vec::new();

        let entry_point = CString::new("main").unwrap();

        for shader in shaders {
            let stage = match shader.stage {
                ShaderStage::Vertex => vk::ShaderStageFlags::VERTEX,
                ShaderStage::Fragment => vk::ShaderStageFlags::FRAGMENT,
                ShaderStage::Geometry => vk::ShaderStageFlags::GEOMETRY,
                _ => unimplemented!(),
            };

            let shader_module_create_info =
                vk::ShaderModuleCreateInfo::default().code(&shader.code);

            let module = match unsafe {
                device_manager
                    .device
                    .create_shader_module(&shader_module_create_info, None)
            } {
                Ok(module) => module,
                Err(e) => {
                    error!("cannot create shader module: {}", e);
                    return Err(GraphicsError::ShaderError);
                }
            };

            let shader_stage_info = ShaderStageInfo {
                device_manager: device_manager.clone(),
                module,
                stage,
                entry_point: entry_point.clone(),
            };

            stages.push(Arc::new(shader_stage_info));
        }

        let binding_descriptions = &[vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(attributes.iter().map(|a| a.size as u32).sum())
            .input_rate(vk::VertexInputRate::VERTEX)];

        let mut attribute_descriptions = vec![];

        for (location, attribute) in zip(0..attributes.len() as u32, attributes) {
            let attribute_description = vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(location)
                .format(match attribute.size {
                    4 => vk::Format::R32_SFLOAT,
                    8 => vk::Format::R32G32_SFLOAT,
                    12 => vk::Format::R32G32B32_SFLOAT,
                    16 => vk::Format::R32G32B32A32_SFLOAT,
                    _ => vk::Format::R32G32B32_SFLOAT,
                })
                .offset(attribute.offset as u32);
            attribute_descriptions.push(attribute_description);
        }

        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(binding_descriptions)
            .vertex_attribute_descriptions(&attribute_descriptions);

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        let viewport = vk::Viewport::default()
            .x(0.)
            .y(0.)
            .width(render_target.extent().width as f32)
            .height(render_target.extent().height as f32)
            .min_depth(0.)
            .max_depth(1.);

        let scissor = vk::Rect2D::default().extent(render_target.extent());

        let dynamic_states = &[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];

        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(dynamic_states);

        let viewports = &[viewport];
        let scissors = &[scissor];

        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewports(viewports)
            .scissors(scissors);

        let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.)
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .depth_bias_enable(false);

        let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
            .sample_shading_enable(false)
            .rasterization_samples(render_target.msaa_samples);

        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
            .alpha_blend_op(vk::BlendOp::ADD);

        let attachments = &[color_blend_attachment];

        let color_blending = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .attachments(attachments);

        let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS)
            .depth_bounds_test_enable(false);

        let stages_vk = Self::stages_as_vk(stages.clone());

        let pipeline_create_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages_vk)
            .vertex_input_state(&vertex_input_info)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisampling)
            .color_blend_state(&color_blending)
            .dynamic_state(&dynamic_state)
            .depth_stencil_state(&depth_stencil_state)
            .layout(layout.pipeline_layout)
            .render_pass(render_target.render_pass);

        match unsafe {
            device_manager.device.create_graphics_pipelines(
                vk::PipelineCache::null(),
                &[pipeline_create_info],
                None,
            )
        } {
            Ok(pipeline) => Ok(Arc::new(Self {
                device_manager,
                layout,
                handle: pipeline[0],
                render_target: Some(render_target),
            })),
            Err(es) => {
                error!("cannot create graphics pipeline: {}", es.1);
                Err(GraphicsError::ShaderError)
            }
        }
    }
}

impl PipelineProxy for VulkanPipeline {
    fn as_vulkan(self: Arc<Self>) -> Option<Arc<VulkanPipeline>> {
        Some(self)
    }
}
