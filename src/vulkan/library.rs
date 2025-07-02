use std::{
    collections::{BTreeMap, HashSet},
    ffi::CStr,
    sync::{Arc, Mutex},
    thread::JoinHandle,
};

use ash::vk::{self, Handle};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

use super::{
    commands::{CommandEntry, CommandManager, CommandType},
    debug_callback::{DebugUtilsMessanger, create_debug_utils_messanger},
    devices::DeviceManager,
    images::VulkanTexture,
    layout,
    memory::BufferManager,
    presentation::Presentation,
    rendering::VulkanRenderTarget,
    validation::get_supported_validation_layers,
};

use crate::{
    GpuSampler, GraphicsApiInitSettings,
    debug::log,
    errors::{CrystalError, CrystalResult},
    images::Image2D,
    object::Object,
    traits::{self, Layout},
    vulkan::{
        VulkanObjectMemoryManager,
        commands::{GpuFuture, GpuSync},
    },
};

pub struct VulkanEntry {
    command_manager: Arc<CommandManager>,
    device_manager: Arc<DeviceManager>,
    _debug_utils_messanger: Option<DebugUtilsMessanger>,

    presentation: Arc<Presentation>,
    thread_handle:
        Mutex<Option<JoinHandle<Result<Box<GpuFuture>, (vk::Result, Arc<Mutex<GpuSync>>)>>>>,

    pub render_targets: BTreeMap<u16, Arc<VulkanRenderTarget>>,
}

impl Drop for VulkanEntry {
    fn drop(&mut self) {
        let graphics = self
            .command_manager
            .command_entries
            .get(&CommandType::Graphics)
            .unwrap();

        let (now, swapchain_out_of_date, _) = self.wait_for_thread(graphics.clone());

        if !swapchain_out_of_date {
            now.acquire_next_image(&self.presentation).unwrap();
        }
    }
}

impl VulkanEntry {
    pub fn with_presentation<T: HasWindowHandle + HasDisplayHandle>(
        settings: &GraphicsApiInitSettings,
        window: &T,
    ) -> CrystalResult<Arc<Self>> {
        let mut instance_extensions = vec![
            #[cfg(debug_assertions)]
            vk::EXT_DEBUG_UTILS_NAME.as_ptr(),
        ];

        let device_extensions = [vk::KHR_SWAPCHAIN_NAME.as_ptr()];

        let mut required_extensions = match ash_window::enumerate_required_extensions(
            window.display_handle().unwrap().as_raw(),
        ) {
            Ok(ext) => ext.to_vec(),
            Err(e) => {
                log!("cannot enumerate required display extensions: {}", e);
                return Err(CrystalError::ConnotInitLibrary);
            }
        };
        instance_extensions.append(&mut required_extensions);

        #[cfg(debug_assertions)]
        unsafe {
            std::env::set_var("VK_LOADER_LAYERS_DISABLE", "~implicit~")
        };

        let entry = match unsafe { ash::Entry::load() } {
            Ok(entry) => Arc::new(entry),
            Err(e) => {
                log!("cannot load vulkan entry: {}", e);
                return CrystalResult::Err(CrystalError::CannotLoadLibrary);
            }
        };

        let app_info = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3);
        let mut create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&instance_extensions);

        #[cfg(debug_assertions)]
        let layers;
        #[cfg(debug_assertions)]
        let layers_pp: Vec<*const i8>;

        #[cfg(debug_assertions)]
        {
            layers = get_supported_validation_layers(&entry);
            if layers.is_empty() {
                log!("no validation layers found!");
                return CrystalResult::Err(CrystalError::CannotCreateDebugMessanger);
            }

            layers_pp = layers.iter().map(|x| x.as_ptr()).collect();

            create_info.pp_enabled_layer_names = layers_pp.as_ptr();
            create_info.enabled_layer_count = layers_pp.len() as u32;
        }

        let instance = match unsafe { entry.create_instance(&create_info, None) } {
            Err(e) => {
                log!("cannot create vulkan instance: {}", e);
                #[cfg(debug_assertions)]
                log!("LAYERS:");

                #[cfg(debug_assertions)]
                layers.iter().for_each(|x| {
                    let layer_bytes = &unsafe { *(x.as_ptr() as *const [u8; 256]) };
                    let layer = CStr::from_bytes_until_nul(layer_bytes)
                        .unwrap()
                        .to_str()
                        .unwrap();
                    log!(" {}", layer);
                });

                return CrystalResult::Err(CrystalError::ConnotInitLibrary);
            }
            Ok(instance) => Arc::new(instance),
        };

        #[cfg(debug_assertions)]
        let debug_utils_messanger = {
            log!("creating debug utils");
            create_debug_utils_messanger(&entry, &instance)?
        };

        let surface = Presentation::create_surface(
            &entry,
            &instance,
            window,
            settings.vsync,
            Some(vk::Extent2D {
                width: settings.width,
                height: settings.height,
            }),
        );

        let device_manager =
            DeviceManager::new(entry, instance, Some(surface.clone()), &device_extensions).unwrap(); // TODO just return

        let command_manager =
            CommandManager::new(device_manager.clone(), settings.double_buffering)?;

        let presentation =
            Presentation::new(device_manager.clone(), surface, settings.msaa_samples)?;

        let viewport_render_target = VulkanRenderTarget::new(
            device_manager.clone(),
            presentation.swapchain.swapchain_info.surface_format.format,
            vk::Extent2D {
                width: settings.width,
                height: settings.height,
            },
            presentation
                .swapchain
                .swapchain_image_views
                .read()
                .unwrap()
                .clone(),
            presentation.msaa_samples,
        )?;

        let mut render_targets = BTreeMap::new();

        render_targets.insert(0, viewport_render_target);

        Ok(Arc::new(Self {
            device_manager,
            command_manager,

            #[cfg(debug_assertions)]
            _debug_utils_messanger: Some(debug_utils_messanger),
            #[cfg(not(debug_assertions))]
            _debug_utils_messanger: None,

            presentation,
            thread_handle: Mutex::new(None),

            render_targets,
        }))
    }

    pub fn recreate_resources(&self, width: u32, height: u32) -> CrystalResult<()> {
        let mut handle_lock = self.thread_handle.lock().unwrap();
        match &*handle_lock {
            Some(_handle) => {
                let _ = handle_lock.take().unwrap().join().unwrap();
                self.command_manager
                    .command_entries
                    .get(&CommandType::Graphics)
                    .clone()
                    .unwrap()
                    .wait()?;
            }
            None => {}
        };

        *handle_lock = None;

        self.presentation
            .swapchain
            .recreate(Some(vk::Extent2D { width, height }))?;
        self.get_viewport().as_vulkan().unwrap().update_resources(
            self.presentation.swapchain.extent(),
            self.presentation
                .swapchain
                .swapchain_image_views
                .read()
                .unwrap()
                .clone(),
        )
    }

    fn wait_for_thread(&self, command_entry: Arc<CommandEntry>) -> (Box<GpuFuture>, bool, bool) {
        let mut swapchain_out_of_date = false;
        let mut suboptimal = false;

        let mut handle_lock = self.thread_handle.lock().unwrap();
        let now = match &*handle_lock {
            Some(_handle) => match handle_lock.take().unwrap().join().unwrap() {
                Ok(n) => n.wait().unwrap(),
                Err((vk::Result::ERROR_OUT_OF_DATE_KHR, sync)) => {
                    swapchain_out_of_date = true;
                    suboptimal = true;
                    command_entry.now_with_sync(sync)
                }
                Err((vk::Result::SUBOPTIMAL_KHR, sync)) => {
                    suboptimal = true;
                    command_entry.now_with_sync(sync)
                }
                Err((e, _)) => {
                    panic!("failed to present queue: {}", e);
                }
            },
            None => command_entry.now(),
        };

        (now, swapchain_out_of_date, suboptimal)
    }

    pub fn render_and_present(self: Arc<Self>, objects: &[Arc<Object>]) -> CrystalResult<()> {
        let graphics = self
            .command_manager
            .command_entries
            .get(&CommandType::Graphics)
            .unwrap()
            .clone();

        let (now, swapchain_out_of_date, suboptimal) = self.wait_for_thread(graphics.clone());

        #[cfg(debug_assertions)]
        self.update_debug_text();

        let render_target_dyn = self.get_viewport();
        let render_target = render_target_dyn
            .clone()
            .as_vulkan()
            .expect("fatal: wrong type of RenderTarget, expected: VulkanRenderTarget");

        if swapchain_out_of_date {
            self.presentation.swapchain.recreate(None)?;
        }

        if suboptimal {
            render_target.update_resources(
                self.presentation.swapchain.extent(),
                self.presentation
                    .swapchain
                    .swapchain_image_views
                    .read()
                    .unwrap()
                    .clone(),
            )?;
        }

        match now.acquire_next_image(&self.presentation) {
            Ok((idx, _)) => *self.presentation.image_index.lock().unwrap() = idx,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.presentation.swapchain.recreate(None)?;
                render_target.update_resources(
                    self.presentation.swapchain.extent(),
                    self.presentation
                        .swapchain
                        .swapchain_image_views
                        .read()
                        .unwrap()
                        .clone(),
                )?;
                return Ok(());
            }
            Err(e) => {
                log!("failed aquire next image: {}", e);
                return Err(CrystalError::RenderingError);
            }
        };

        let n_pass = now.n_pass();

        let color = 0.2f32;
        let mut clear_color = vk::ClearColorValue::default();
        let clear_depth_stencil = vk::ClearDepthStencilValue::default().depth(1.).stencil(0);
        unsafe {
            clear_color.float32[0] = color;
            clear_color.float32[1] = color;
            clear_color.float32[2] = color;
            clear_color.float32[3] = 1.0f32
        };
        let clear_value_color = vk::ClearValue { color: clear_color };
        let clear_value_stencil = vk::ClearValue {
            depth_stencil: clear_depth_stencil,
        };
        let clear_values = &[clear_value_color, clear_value_stencil];

        let gpu_future = now.join(graphics.clone().record_command_buffer(
            n_pass,
            |command_buffer, device| {
                let render_pass_begin = vk::RenderPassBeginInfo::default()
                    .render_pass(render_target.render_pass)
                    .framebuffer(
                        *render_target.framebuffers
                            [*self.presentation.image_index.lock().unwrap() as usize]
                            .read()
                            .unwrap(),
                    )
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D::default().x(0).y(0),
                        extent: vk::Extent2D {
                            width: render_target
                                .extent()
                                .width
                                .min(self.presentation.swapchain.extent().width),
                            height: render_target
                                .extent()
                                .height
                                .min(self.presentation.swapchain.extent().height),
                        }, // render_target.extent,
                    })
                    .clear_values(clear_values);

                unsafe {
                    device.cmd_begin_render_pass(
                        *command_buffer,
                        &render_pass_begin,
                        vk::SubpassContents::INLINE,
                    )
                }

                let viewport = vk::Viewport::default()
                    .width(render_target.extent().width as f32)
                    .height(render_target.extent().height as f32)
                    .max_depth(1.);
                let viewports = &[viewport];
                unsafe { device.cmd_set_viewport(*command_buffer, 0, viewports) }
                let scissor = vk::Rect2D::default().extent(render_target.extent());
                let scissors = &[scissor];
                unsafe { device.cmd_set_scissor(*command_buffer, 0, scissors) }

                let layouts: HashSet<_> = objects
                    .iter()
                    .map(|object| object.pipeline.clone().as_vulkan().unwrap().layout.clone())
                    .collect();

                // TODO optimize
                for layout in layouts {
                    let objects: Vec<Arc<Object>> = objects
                        .iter()
                        .filter_map(|object| {
                            if object
                                .pipeline
                                .clone()
                                .as_vulkan()
                                .unwrap()
                                .layout
                                .pipeline_layout
                                .as_raw()
                                == layout.pipeline_layout.as_raw()
                            {
                                Some(object.clone())
                            } else {
                                None
                            }
                        })
                        .collect();

                    layout.render(&objects, command_buffer).unwrap();
                }

                unsafe { device.cmd_end_render_pass(*command_buffer) }
            },
        )?);

        let presentation = self.presentation.clone();

        let handle =
            std::thread::spawn(move || gpu_future.then_swapchain_present_and_flush(presentation));

        let mut handle_lock = self.thread_handle.lock().unwrap();
        *handle_lock = Some(handle);

        Ok(())
    }

    pub fn create_layout(
        &self,
        texture_num: usize,
        sampler_num: usize,
        uniform_num: usize,
        storage_num: usize,
    ) -> CrystalResult<Arc<dyn Layout>> {
        Ok(layout::VulkanLayout::new(
            self.device_manager.clone(),
            texture_num,
            sampler_num,
            uniform_num,
            storage_num,
            self.command_manager
                .command_entries
                .get(&CommandType::Graphics)
                .clone()
                .unwrap()
                .double_buffering,
        )?)
    }

    pub fn create_sampler(
        &self,
        image: &Image2D,
        anisotropy_texels: f32,
    ) -> CrystalResult<Arc<GpuSampler>> {
        let texture = VulkanTexture::new(
            self.device_manager.clone(),
            image,
            self.command_manager.clone(),
            anisotropy_texels,
        )?;
        Ok(GpuSampler::from_texture(texture))
    }

    pub fn register_meshes(&self, objects: &[Arc<Object>]) {
        objects.iter().for_each(|object| {
            let mut memory_manager = object.memory_manager.write().unwrap();

            match object.mesh.clone() {
                Some(mesh) => match *memory_manager {
                    Some(_) => {}
                    None => {
                        *memory_manager = Some(
                            VulkanObjectMemoryManager::new(
                                self.device_manager.clone(),
                                &mesh.vertices,
                                &mesh.indices,
                            )
                            .unwrap(),
                        );
                    }
                },
                None => {}
            }
        });
    }

    pub fn get_viewport(&self) -> Arc<dyn traits::RenderTarget> {
        self.render_targets[&0].clone()
    }

    pub fn get_raw_device_handle(&self) -> u64 {
        self.device_manager.device.handle().as_raw()
    }

    pub fn update_debug_text(&self) -> String {
        let mut budget_props = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
        let mut mem_props =
            vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget_props);
        unsafe {
            self.device_manager
                .instance
                .get_physical_device_memory_properties2(
                    self.device_manager.physical_device,
                    &mut mem_props,
                )
        }

        let swapchain_images = unsafe {
            self.presentation
                .swapchain
                .swapchain
                .read()
                .unwrap()
                .get_swapchain_images(*self.presentation.swapchain.swapchain_khr.read().unwrap())
        }
        .unwrap();

        let mut swapchain_memory_usage = 0;

        for image in swapchain_images {
            let mem = unsafe {
                self.device_manager
                    .device
                    .get_image_memory_requirements(image)
            };

            swapchain_memory_usage += mem.size;
        }

        let entries = [
            (
                "budget",
                budget_props.heap_budget[0] as f32 / 1024f32 / 1024f32,
            ),
            (
                "heap",
                budget_props.heap_usage[0] as f32 / 1024f32 / 1024f32,
            ),
            (
                "swapchain",
                swapchain_memory_usage as f32 / 1024f32 / 1024f32,
            ),
        ];

        let mut max_len = entries[0].0.len();

        entries.iter().for_each(|(name, _)| {
            if max_len < name.len() {
                max_len = name.len()
            }
        });

        let formatted: String = entries
            .iter()
            .map(|entry| {
                format!(
                    "{} {:.1} MB\n",
                    format!("{}:{}", entry.0, " ".repeat(max_len - entry.0.len())),
                    entry.1
                )
            })
            .collect();

        formatted
    }

    pub fn write_buffer_to_screen(
        &self,
        buffer: Vec<u8>,
        coords: (u32, u32),
        size: (u32, u32),
    ) -> CrystalResult<()> {
        let staging_buffer = BufferManager::new(
            self.device_manager.clone(),
            buffer.len() as u64,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
        )
        .unwrap();

        staging_buffer.single_time_write(&buffer, 0).unwrap();

        let swapchain_images = unsafe {
            self.presentation
                .swapchain
                .swapchain
                .read()
                .unwrap()
                .get_swapchain_images(*self.presentation.swapchain.swapchain_khr.read().unwrap())
        }
        .unwrap();

        let swapchain_image =
            swapchain_images[*self.presentation.image_index.lock().unwrap() as usize];

        let command_entry = self
            .command_manager
            .command_entries
            .get(&CommandType::Transfer)
            .clone()
            .unwrap();

        let future = command_entry
            .record_single_time_buffer(|command_buffer, device| {
                let subresource_range = vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                };

                // Transition image layout for transfer
                let barrier = vk::ImageMemoryBarrier::default()
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .image(swapchain_image)
                    .subresource_range(subresource_range)
                    .src_access_mask(vk::AccessFlags::empty())
                    .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);

                unsafe {
                    device.cmd_pipeline_barrier(
                        *command_buffer,
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[barrier],
                    )
                };

                // Copy buffer to image
                let region = vk::BufferImageCopy::default()
                    .buffer_offset(0)
                    .buffer_row_length(0)
                    .buffer_image_height(0)
                    .image_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .image_offset(vk::Offset3D {
                        x: coords.0 as i32,
                        y: coords.1 as i32,
                        z: 0,
                    })
                    .image_extent(vk::Extent3D {
                        width: size.0,
                        height: size.1,
                        depth: 1,
                    });

                unsafe {
                    device.cmd_copy_buffer_to_image(
                        *command_buffer,
                        staging_buffer.buffer,
                        swapchain_image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &[region],
                    )
                };

                // Transition back for presentation
                let barrier = vk::ImageMemoryBarrier::default()
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                    .image(swapchain_image)
                    .subresource_range(subresource_range)
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::MEMORY_READ);

                unsafe {
                    device.cmd_pipeline_barrier(
                        *command_buffer,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[barrier],
                    )
                };
            })
            .unwrap();

        future.flush().unwrap();

        command_entry.wait().unwrap();

        Ok(())
    }
}
