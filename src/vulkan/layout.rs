use std::{
    cell::RefCell,
    collections::{BTreeMap, VecDeque},
    sync::{Arc, RwLock},
};

use ash::vk;

use crate::{
    GpuVec,
    debug::log,
    errors::{CrystalError, CrystalResult},
    object::Object,
    traits,
};

use super::{
    devices::DeviceManager, images::VulkanTexture, memory::BufferManager,
    memory_obj::VulkanObjectMemoryManager,
};

pub struct VulkanLayout {
    device_manager: Arc<DeviceManager>,

    descriptor_pool: vk::DescriptorPool,
    descriptor_set_layouts: Vec<vk::DescriptorSetLayout>,

    uniform_buffer_managers_sets: Vec<Vec<Arc<BufferManager>>>,
    storage_buffer_managers_sets: Vec<Vec<Arc<BufferManager>>>,

    uniform_descriptor_sets: Vec<vk::DescriptorSet>,
    storage_descriptor_sets: Vec<vk::DescriptorSet>,
    sampler_descriptor_sets: Vec<vk::DescriptorSet>,

    pub pipeline_layout: vk::PipelineLayout,

    image_views_sampled_num: u32,
    max_instance_num: u64,

    sampler_binding_data: RwLock<BTreeMap<u64, usize>>,
    sampler_binding_data_pool: RwLock<Vec<u64>>,
    samplers: RwLock<BTreeMap<u32, vk::Sampler>>,
    object_render_queue: RwLock<VecDeque<Arc<RefCell<Object>>>>,
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

            self.samplers
                .read()
                .unwrap()
                .iter()
                .for_each(|(_, &sampler)| {
                    self.device_manager.device.destroy_sampler(sampler, None)
                });

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

    fn add_object_to_queue(&self, object: Arc<RefCell<Object>>) {
        self.object_render_queue
            .try_write()
            .unwrap()
            .push_back(object);
    }

    fn write_to_buffer(
        &self,
        is_uniform: bool,
        frame: usize,
        buffer: usize,
        offset: usize,
        data: GpuVec,
    ) -> CrystalResult<()> {
        if is_uniform {
            self.uniform_buffer_managers_sets[frame][buffer].write(data.as_words(), offset)
        } else {
            self.storage_buffer_managers_sets[frame][buffer].write(data.as_words(), offset)
        }
    }
}

impl VulkanLayout {
    pub(crate) fn new(
        device_manager: Arc<DeviceManager>,
        image_views_sampled_num: u32,
        frames_in_flight: u32,
        max_instance_num: u64,

        buffers: &[(bool, u64)],
    ) -> CrystalResult<Arc<Self>> {
        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .descriptor_count(frames_in_flight)
                .ty(vk::DescriptorType::UNIFORM_BUFFER),
            vk::DescriptorPoolSize::default()
                .descriptor_count(frames_in_flight)
                .ty(vk::DescriptorType::STORAGE_BUFFER),
            vk::DescriptorPoolSize::default()
                .descriptor_count(frames_in_flight)
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER),
        ];

        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(frames_in_flight * (2 + max_instance_num as u32));

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

        let mut ubo_bindings = vec![];
        let mut ssbo_bindings = vec![];

        for buffer in buffers {
            if buffer.0 {
                ubo_bindings.push(
                    vk::DescriptorSetLayoutBinding::default()
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                        .stage_flags(vk::ShaderStageFlags::ALL_GRAPHICS)
                        .descriptor_count(1),
                )
            } else {
                ssbo_bindings.push(
                    vk::DescriptorSetLayoutBinding::default()
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .stage_flags(vk::ShaderStageFlags::ALL_GRAPHICS)
                        .descriptor_count(1),
                )
            }
        }

        let ubo_bindings_count = ubo_bindings.len() as u32;
        for idx in 0..ubo_bindings_count {
            ubo_bindings[idx as usize].descriptor_count = ubo_bindings_count;
            ubo_bindings[idx as usize].binding = idx as u32;
        }

        let ssbo_bindings_count = ssbo_bindings.len() as u32;
        for idx in 0..ssbo_bindings_count {
            ssbo_bindings[idx as usize].descriptor_count = ssbo_bindings_count;
            ssbo_bindings[idx as usize].binding = idx as u32;
        }

        let mut sampler_bindings = vec![];

        for i in 0..image_views_sampled_num {
            sampler_bindings.push(
                vk::DescriptorSetLayoutBinding::default()
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                    .descriptor_count(image_views_sampled_num)
                    .binding(i),
            )
        }

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

        let ubo_layouts = vec![ubo_descriptor_set_layout; frames_in_flight as usize];
        let ssbo_layouts = vec![ssbo_descriptor_set_layout; frames_in_flight as usize];
        let sampler_layouts = vec![
            sampler_descriptor_set_layout;
            frames_in_flight as usize * max_instance_num as usize
        ];

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

        let mut uniform_buffer_managers_sets = vec![];
        let mut storage_buffer_managers_sets = vec![];

        for i in 0..frames_in_flight as usize {
            let mut uniform_buffer_managers = vec![];
            let mut storage_buffer_managers = vec![];

            let mut current_uniform = 0;
            let mut current_storage = 0;

            for buffer in buffers {
                let size = buffer.1;

                if buffer.0 {
                    let uniform_buffer_manager = BufferManager::new(
                        device_manager.clone(),
                        size,
                        vk::BufferUsageFlags::UNIFORM_BUFFER,
                        vk::MemoryPropertyFlags::HOST_VISIBLE
                            | vk::MemoryPropertyFlags::HOST_COHERENT,
                    )?;

                    uniform_buffer_manager.map_memory(size, 0)?;

                    let uniform_buffer_info = vk::DescriptorBufferInfo::default()
                        .buffer(uniform_buffer_manager.buffer)
                        .offset(0)
                        .range(size);

                    uniform_buffer_managers.push(uniform_buffer_manager);

                    let buffer_infos = &[uniform_buffer_info];

                    let descriptor_write = vk::WriteDescriptorSet::default()
                        .dst_set(uniform_descriptor_sets[i])
                        .dst_binding(current_uniform)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                        .descriptor_count(1)
                        .buffer_info(buffer_infos);

                    unsafe {
                        device_manager
                            .device
                            .update_descriptor_sets(&[descriptor_write], &[])
                    };

                    current_uniform += 1;
                } else {
                    let storage_buffer_manager = BufferManager::new(
                        device_manager.clone(),
                        size,
                        vk::BufferUsageFlags::STORAGE_BUFFER,
                        vk::MemoryPropertyFlags::HOST_VISIBLE
                            | vk::MemoryPropertyFlags::HOST_COHERENT,
                    )?;

                    storage_buffer_manager.map_memory(size, 0)?;

                    let storage_buffer_info = vk::DescriptorBufferInfo::default()
                        .buffer(storage_buffer_manager.buffer)
                        .offset(0)
                        .range(size);

                    storage_buffer_managers.push(storage_buffer_manager);

                    let buffer_infos = &[storage_buffer_info];

                    let descriptor_write = vk::WriteDescriptorSet::default()
                        .dst_set(storage_descriptor_sets[i])
                        .dst_binding(current_storage)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .buffer_info(buffer_infos);

                    unsafe {
                        device_manager
                            .device
                            .update_descriptor_sets(&[descriptor_write], &[])
                    };

                    current_storage += 1;
                }
            }

            uniform_buffer_managers_sets.push(uniform_buffer_managers);
            storage_buffer_managers_sets.push(storage_buffer_managers);
        }

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
            device_manager,

            descriptor_pool,
            descriptor_set_layouts,

            uniform_buffer_managers_sets,
            storage_buffer_managers_sets,

            uniform_descriptor_sets,
            storage_descriptor_sets,
            sampler_descriptor_sets,

            image_views_sampled_num,
            max_instance_num,

            pipeline_layout,

            sampler_binding_data: RwLock::new(BTreeMap::new()),
            sampler_binding_data_pool: RwLock::new(vec![]),
            samplers: RwLock::new(BTreeMap::new()),
            object_render_queue: RwLock::new(VecDeque::new()),
        }))
    }

    pub(crate) fn render(
        &self,
        device_manager: Arc<DeviceManager>,
        command_buffer: &vk::CommandBuffer,
        frames_in_flight: usize,
        frame: usize,
    ) -> CrystalResult<()> {
        unsafe {
            device_manager.device.cmd_bind_descriptor_sets(
                *command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &[
                    self.uniform_descriptor_sets[frame],
                    self.storage_descriptor_sets[frame],
                ],
                &[],
            )
        }

        let mut current_object_idx = 0usize;

        while self.object_render_queue.read().unwrap().len() > 0 {
            let obj = self
                .object_render_queue
                .try_write()
                .unwrap()
                .pop_front()
                .unwrap();
            let mut obj = obj.borrow_mut();

            match obj.mesh.clone() {
                Some(mesh) => match obj.memory_manager {
                    Some(_) => {}
                    None => {
                        obj.memory_manager = Some(
                            VulkanObjectMemoryManager::new(
                                device_manager.clone(),
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
                    let texture_sets: Vec<Arc<VulkanTexture>> = textures
                        .iter()
                        .map(|texture| match texture.1.clone().as_vulkan() {
                            Some(tex) => tex,
                            None => panic!("fatal: wrong type of textures, expected vulkan"),
                        })
                        .collect();

                    let sets = self.init_sampler_descriptor_sets(
                        device_manager.clone(),
                        frames_in_flight,
                        textures.as_ptr() as usize as u64,
                        &texture_sets,
                    )?;

                    unsafe {
                        device_manager.device.cmd_bind_descriptor_sets(
                            *command_buffer,
                            vk::PipelineBindPoint::GRAPHICS,
                            self.pipeline_layout,
                            2,
                            &[sets[frame]],
                            &[],
                        )
                    }
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

            let index_buffer = object_memory_manager.index_buffer_manager.buffer;
            let vertex_buffer = object_memory_manager.vertex_buffer_manager.buffer;

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

            let index_count = obj.mesh.as_ref().unwrap().indices.len();

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

            current_object_idx += 1;
        }

        self.release_sampler_descriptor_sets();
        Ok(())
    }

    fn get_sampler(
        &self,
        device_manager: Arc<DeviceManager>,
        mip_levels: u32,
        anisotropy_texels: f32,
    ) -> CrystalResult<vk::Sampler> {
        match self.samplers.read().unwrap().get(&mip_levels) {
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

        let sampler = match unsafe { device_manager.device.create_sampler(&sampler_info, None) } {
            Ok(sampler) => sampler,
            Err(e) => {
                log!("cannot create sampler: {}", e);
                return Err(CrystalError::ImageError);
            }
        };

        self.samplers
            .try_write()
            .unwrap()
            .insert(mip_levels, sampler);

        Ok(sampler)
    }

    fn release_sampler_descriptor_sets(&self) {
        let mut to_delete_sets = vec![];

        for &slot in self.sampler_binding_data.read().unwrap().keys() {
            if self
                .sampler_binding_data_pool
                .read()
                .unwrap()
                .iter()
                .find(|&&x| x == slot)
                .is_none()
            {
                to_delete_sets.push(slot);
            }
        }

        self.sampler_binding_data_pool.try_write().unwrap().clear();

        for to_delete in to_delete_sets {
            self.sampler_binding_data
                .try_write()
                .unwrap()
                .remove(&to_delete)
                .unwrap();
        }
    }

    pub(crate) fn init_sampler_descriptor_sets(
        &self,
        device_manager: Arc<DeviceManager>,
        frames_in_flight: usize,
        object_id: u64,
        textures_set: &[Arc<VulkanTexture>],
    ) -> CrystalResult<Vec<vk::DescriptorSet>> {
        let mut sampler_binding_data = self.sampler_binding_data.write().unwrap();
        let sampler = sampler_binding_data.get(&object_id);

        let texture_set_idx = match sampler {
            Some(&idx) => {
                let sets = self.sampler_descriptor_sets
                    [frames_in_flight as usize * idx..frames_in_flight as usize * (idx + 1)]
                    .to_vec();
                self.sampler_binding_data_pool
                    .try_write()
                    .unwrap()
                    .push(object_id);
                return Ok(sets);
            }
            None => {
                let mut idx = 0usize;
                while sampler_binding_data
                    .iter()
                    .find(|binding| *binding.1 == idx)
                    .is_some()
                {
                    idx += 1;
                }
                if idx >= self.max_instance_num as usize {
                    log!("Exceeded number of maximum image number");
                    return Err(CrystalError::DescriptorError);
                }
                sampler_binding_data.insert(object_id, idx);
                self.sampler_binding_data_pool
                    .try_write()
                    .unwrap()
                    .push(object_id);
                idx
            }
        };

        let sets = self.sampler_descriptor_sets[frames_in_flight as usize * texture_set_idx
            ..frames_in_flight as usize * (texture_set_idx + 1)]
            .to_vec();

        for current_frame in 0..frames_in_flight {
            let mut image_info = vec![];

            assert!(
                (textures_set.len() as u32) <= self.image_views_sampled_num,
                "fatal: the texture number limit for layout is exaggerated"
            );

            for texture in textures_set {
                let sampler = self.get_sampler(
                    device_manager.clone(),
                    texture.image.mip_levels,
                    texture.image.anisotropy_texels,
                )?;

                let descriptor_image_info = vk::DescriptorImageInfo::default()
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image_view(texture.image.image_view)
                    .sampler(sampler);

                image_info.push(descriptor_image_info);
            }

            let mut descriptor_writes = vec![];

            for image_view_idx in 0..textures_set.len() {
                let sampler_descriptor_write = vk::WriteDescriptorSet::default()
                    .dst_set(sets[current_frame])
                    .dst_binding(image_view_idx as u32)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&image_info);

                descriptor_writes.push(sampler_descriptor_write);
            }

            unsafe {
                device_manager
                    .device
                    .update_descriptor_sets(&descriptor_writes, &[])
            };
        }

        Ok(sets)
    }
}
