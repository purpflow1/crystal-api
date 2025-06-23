use std::{
    collections::{BTreeMap, VecDeque},
    hash::Hash,
    sync::{Arc, Mutex, MutexGuard, RwLock},
    thread::JoinHandle,
};

use ash::vk::{self, Handle};
use pollster::FutureExt;

use crate::{
    GpuSampler,
    debug::log,
    errors::{CrystalError, CrystalResult},
    gpu_data::{IntoGpuBuffer, IntoGpuTexture, PtrHandler},
    object::Object,
    traits,
    vulkan::VulkanTexture,
};

use super::{devices::DeviceManager, memory::BufferManager};

struct LayoutDynamicData {
    device_manager: Arc<DeviceManager>,

    uniform_descriptor_sets: Vec<vk::DescriptorSet>,
    storage_descriptor_sets: Vec<vk::DescriptorSet>,
    sampler_descriptor_sets: Vec<vk::DescriptorSet>,

    n_pass: usize,
    double_buffering: bool,

    sampler_binding_data: BTreeMap<usize, (Arc<VulkanTexture>, Arc<RwLock<bool>>)>,
    buffer_managers_sets: BTreeMap<usize, (Vec<Arc<BufferManager>>, Arc<dyn IntoGpuBuffer>)>,
    samplers: BTreeMap<u32, vk::Sampler>,

    thread_handle: Option<JoinHandle<()>>,
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
    fn update_data(&mut self) {
        if let Some(_handle) = &self.thread_handle {
            self.thread_handle.take().unwrap().join().unwrap();
        }

        let indices_to_clean: Vec<usize> = self
            .sampler_binding_data
            .iter()
            .filter(|(_, (_, alive))| !*alive.read().unwrap())
            .map(|(ind, _)| *ind)
            .collect();

        for ind in indices_to_clean {
            self.sampler_binding_data.remove(&ind);
        }

        let mut buffer_tasks = VecDeque::new();

        let indices_to_clean: Vec<usize> = self
            .buffer_managers_sets
            .iter()
            .filter(|(_, (buffers, gpu_buffer))| {
                let ptr = gpu_buffer.get_ptr();
                let mut ptr = ptr.0.write().unwrap();

                if ptr.is_null() {
                    true
                } else {
                    for task in gpu_buffer.query_tasks() {
                        *ptr = (*buffers[self.n_pass].mapped_memory.read().unwrap()).unwrap()
                            as *mut u8;

                        buffer_tasks.push_back((task, PtrHandler(RwLock::new(*ptr))));
                    }
                    false
                }
            })
            .map(|(ind, _)| *ind)
            .collect();

        for ind in indices_to_clean {
            self.buffer_managers_sets.remove(&ind);
        }

        if self.double_buffering {
            self.n_pass = (self.n_pass + 1) % 2;
        };

        let handle = std::thread::spawn(move || {
            while let Some((task, ptr)) = buffer_tasks.pop_front() {
                task.flush((*ptr.0.read().unwrap()) as *mut u8).block_on();
            }
        });

        self.thread_handle = Some(handle);
    }

    fn add_textures(
        &mut self,
        descriptor_set_id: &mut usize,
        samplers: &[(u32, Arc<GpuSampler>)],
    ) -> CrystalResult<()> {
        for (binding, texture) in samplers {
            let id = (0..usize::MAX)
                .find(|idx| {
                    self.sampler_binding_data
                        .iter()
                        .find(|(x, _)| **x == *idx)
                        .is_none()
                })
                .unwrap();

            let vulkan_texture = texture.get_texture().as_vulkan().unwrap();
            *descriptor_set_id = id;

            self.sampler_binding_data
                .insert(id, (vulkan_texture.clone(), texture.get_alive()));

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
                            log!("cannot create sampler: {}", e);
                            return Err(CrystalError::ImageError);
                        }
                    }
                }
            };

            let image_infos = [vk::DescriptorImageInfo::default()
                .image_view(vulkan_texture.image.image_view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .sampler(sampler)];

            let descriptor_write = {
                let descriptor_set = self.sampler_descriptor_sets[*descriptor_set_id];

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

    fn add_buffer(
        &mut self,
        binding: usize,
        is_uniform: bool,
        data: Arc<dyn crate::gpu_data::IntoGpuBuffer>,
    ) -> CrystalResult<()> {
        let size = data.size() as u64;

        let (descriptor_type, buffer_usage) = if is_uniform {
            (
                vk::DescriptorType::UNIFORM_BUFFER,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
            )
        } else {
            (
                vk::DescriptorType::STORAGE_BUFFER,
                vk::BufferUsageFlags::STORAGE_BUFFER,
            )
        };

        let idx = (0..usize::MAX)
            .find(|idx| {
                self.buffer_managers_sets
                    .iter()
                    .find(|(x, _)| **x == *idx)
                    .is_none()
            })
            .unwrap();

        let mut buffers = vec![];

        for buffer_idx in 0..if self.double_buffering { 2 } else { 1 } {
            let buffer_manager = BufferManager::new(
                self.device_manager.clone(),
                size,
                buffer_usage,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;

            let ptr = buffer_manager.map_memory(size, 0)?;
            buffers.push(buffer_manager.clone());

            *data.get_ptr().0.write().unwrap() = ptr;

            let uniform_buffer_info = vk::DescriptorBufferInfo::default()
                .buffer(buffer_manager.buffer)
                .offset(0)
                .range(size);

            let buffer_infos = &[uniform_buffer_info];

            let descriptor_set = if is_uniform {
                self.uniform_descriptor_sets[buffer_idx]
            } else {
                self.storage_descriptor_sets[buffer_idx]
            };

            let descriptor_write = vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(binding as u32)
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

        let tasks = data.query_tasks();

        if buffers.len() > 1 {
            let ptr = (*buffers[1].mapped_memory.read().unwrap()).unwrap() as *mut u8;

            for task in tasks.clone() {
                task.flush(ptr).block_on();
            }
        }

        let ptr = (*buffers[0].mapped_memory.read().unwrap()).unwrap() as *mut u8;

        for task in tasks {
            task.flush(ptr).block_on();
        }

        let to_push = (buffers, data);
        self.buffer_managers_sets.insert(idx, to_push);

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

impl PartialEq for VulkanLayout {
    fn eq(&self, other: &Self) -> bool {
        self.pipeline_layout.as_raw() == other.pipeline_layout.as_raw()
    }
}
impl Eq for VulkanLayout {}
impl Hash for VulkanLayout {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.pipeline_layout.as_raw().hash(state);
    }
}

impl Drop for VulkanLayout {
    fn drop(&mut self) {
        unsafe {
            // descriptorPool must have been created with the VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT flag
            //
            // let descriptor_sets: Vec<vk::DescriptorSet> = self
            //     .uniform_descriptor_sets
            //     .iter()
            //     .chain(self.storage_descriptor_sets.iter())
            //     .chain(self.sampler_descriptor_sets.iter())
            //     .map(|&descriptor_set| descriptor_set)
            //     .collect();

            // self.device_manager
            //     .device
            //     .free_descriptor_sets(self.descriptor_pool, &descriptor_sets)
            //     .unwrap();

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

impl traits::Layout for VulkanLayout {
    fn as_vulkan(self: Arc<Self>) -> Option<Arc<super::VulkanLayout>> {
        Some(self)
    }

    fn add_buffer(
        &self,
        binding: usize,
        is_uniform: bool,
        data: Arc<dyn crate::gpu_data::IntoGpuBuffer>,
    ) -> CrystalResult<()> {
        let mut dynamic_data = self.dynamic_data.lock().unwrap();
        dynamic_data.add_buffer(binding, is_uniform, data)
    }

    fn register_samplers(&self, objects: &[Arc<Object>]) -> CrystalResult<()> {
        let mut dynamic_data = self.dynamic_data.lock().unwrap();

        objects.iter().for_each(|object| {
            dynamic_data
                .add_textures(
                    &mut *object.id.lock().unwrap(),
                    object.samplers.as_ref().expect("object has no samplers!"),
                )
                .unwrap()
        });
        Ok(())
    }
}

impl VulkanLayout {
    pub(crate) fn new(
        device_manager: Arc<DeviceManager>,
        texture_num: usize,
        sampler_num: usize,
        uniform_num: usize,
        storage_num: usize,

        double_buffering: bool,
    ) -> CrystalResult<Arc<Self>> {
        let buffer_count = if double_buffering { 2 } else { 1 };

        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .descriptor_count(buffer_count * uniform_num as u32)
                .ty(vk::DescriptorType::UNIFORM_BUFFER),
            vk::DescriptorPoolSize::default()
                .descriptor_count(buffer_count * storage_num as u32)
                .ty(vk::DescriptorType::STORAGE_BUFFER),
            vk::DescriptorPoolSize::default()
                .descriptor_count((sampler_num * texture_num) as u32)
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER),
        ];

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
                log!("cannot create descriptor pool: {}", e);
                return Err(CrystalError::DescriptorError);
            }
        };

        let ubo_bindings: Vec<_> = (0..uniform_num)
            .map(|idx| {
                vk::DescriptorSetLayoutBinding::default()
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .stage_flags(vk::ShaderStageFlags::ALL_GRAPHICS)
                    .descriptor_count(1 as u32)
                    .binding(idx as u32)
            })
            .collect();

        let ssbo_bindings: Vec<_> = (0..storage_num)
            .map(|idx| {
                vk::DescriptorSetLayoutBinding::default()
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .stage_flags(vk::ShaderStageFlags::ALL_GRAPHICS)
                    .descriptor_count(1 as u32)
                    .binding(idx as u32)
            })
            .collect();

        let sampler_bindings: Vec<_> = (0..texture_num)
            .map(|idx| {
                vk::DescriptorSetLayoutBinding::default()
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                    .descriptor_count(1 as u32)
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
                log!("cannot create descriptor set layout: {}", e);
                return Err(CrystalError::DescriptorError);
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
                log!("cannot create descriptor set layout: {}", e);
                return Err(CrystalError::DescriptorError);
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
                log!("cannot create descriptor set layout: {}", e);
                return Err(CrystalError::DescriptorError);
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

        let uniform_descriptor_sets = match unsafe {
            device_manager
                .device
                .allocate_descriptor_sets(&ubo_alloc_info)
        } {
            Ok(descriptor_sets) => descriptor_sets,
            Err(e) => {
                log!("cannot allocate uniform descriptor sets: {}", e);
                return Err(CrystalError::DescriptorError);
            }
        };

        let storage_descriptor_sets = match unsafe {
            device_manager
                .device
                .allocate_descriptor_sets(&ssbo_alloc_info)
        } {
            Ok(descriptor_sets) => descriptor_sets,
            Err(e) => {
                log!("cannot allocate storage descriptor sets: {}", e);
                return Err(CrystalError::DescriptorError);
            }
        };

        let sampler_descriptor_sets = match unsafe {
            device_manager
                .device
                .allocate_descriptor_sets(&sampler_alloc_info)
        } {
            Ok(descriptor_sets) => descriptor_sets,
            Err(e) => {
                log!("cannot allocate sampler descriptor sets: {}", e);
                return Err(CrystalError::DescriptorError);
            }
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
                log!("cannot create pipeline layout: {}", e);
                return Err(CrystalError::DescriptorError);
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

                buffer_managers_sets: BTreeMap::new(),
                samplers: BTreeMap::new(),
                thread_handle: None,
            }),
        }))
    }

    pub(crate) fn render(
        &self,
        objects: &[Arc<Object>],
        command_buffer: &vk::CommandBuffer,
    ) -> CrystalResult<()> {
        let mut dynamic_data = self.dynamic_data.lock().unwrap();
        let device_manager = dynamic_data.device_manager.clone();

        dynamic_data.update_data();

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

        for (current_object_idx, object) in objects.iter().enumerate() {
            if let Some(_samplers) = &object.samplers {
                let id = *object.id.lock().unwrap();

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
                None => panic!("fatal: wrong pipeline type, expected vulkan"),
            };

            let memory_manager = object.memory_manager.read().unwrap();

            let vulkan_memory_manager = memory_manager
                .as_ref()
                .expect("object has not registered to api!")
                .as_vulkan_ref()
                .unwrap();

            let index_buffer = vulkan_memory_manager.index_buffer_manager.buffer;
            let vertex_buffer = vulkan_memory_manager.vertex_buffer_manager.buffer;

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

            let index_count = object.mesh.as_ref().unwrap().indices.len();

            unsafe {
                device_manager.device.cmd_bind_vertex_buffers(
                    *command_buffer,
                    0,
                    &[vertex_buffer],
                    &[0],
                )
            }

            unsafe {
                device_manager.device.cmd_draw_indexed(
                    *command_buffer,
                    index_count as u32,
                    1,
                    0 as u32,
                    0 as i32,
                    current_object_idx as u32,
                )
            }
        }

        Ok(())
    }

    fn get_sampler(
        &self,
        dynamic_data: &mut MutexGuard<LayoutDynamicData>,
        mip_levels: u32,
        anisotropy_texels: f32,
    ) -> CrystalResult<vk::Sampler> {
        match dynamic_data.samplers.get(&mip_levels) {
            Some(sampler) => return Ok(*sampler),
            None => (),
        };

        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::REPEAT)
            .address_mode_v(vk::SamplerAddressMode::REPEAT)
            .address_mode_w(vk::SamplerAddressMode::REPEAT)
            .anisotropy_enable(true)
            .anisotropy_enable(anisotropy_texels > 1.)
            .max_anisotropy(anisotropy_texels)
            .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
            .unnormalized_coordinates(false)
            .compare_enable(false)
            .compare_op(vk::CompareOp::ALWAYS)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .mip_lod_bias(0.)
            .min_lod(0.)
            .max_lod(mip_levels as f32);

        let sampler = match unsafe {
            self.device_manager
                .device
                .create_sampler(&sampler_info, None)
        } {
            Ok(sampler) => sampler,
            Err(e) => {
                log!("cannot create sampler: {}", e);
                return Err(CrystalError::ImageError);
            }
        };

        dynamic_data.samplers.insert(mip_levels, sampler);

        Ok(sampler)
    }
}
