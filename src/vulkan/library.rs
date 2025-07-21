use std::{
    collections::BTreeMap,
    ffi::CStr,
    sync::{Arc, Mutex},
};

use ash::vk::{self, Handle};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

use super::{
    commands::{CommandManager, CommandType, PresentResult},
    debug_callback::{DebugUtilsMessanger, create_debug_utils_messanger},
    devices::DeviceManager,
    images::VulkanTexture,
    layout,
    memory::{BufferInfo, BufferManager},
    presentation::Presentation,
    rendering::VulkanRenderTarget,
    sync::GpuSync,
    validation::get_supported_validation_layers,
};

use crate::{
    AsBytes, Buffer, GpuSampler, GraphicsApiInitSettings,
    debug::log,
    errors::{CrystalError, CrystalResult},
    mesh::{Index, Mesh, VertexTexture},
    object::{MeshBuffer, Object},
    settings::DebugData,
    traits::{self, Layout},
    vulkan::VulkanLayout,
};

pub struct TimeState {
    timer: std::time::Instant,
    delta_time: std::time::Duration,
}

pub struct VulkanEntry {
    command_manager: Arc<CommandManager>,
    device_manager: Arc<DeviceManager>,
    _debug_utils_messanger: Option<DebugUtilsMessanger>,

    presentation: Arc<Presentation>,

    present_result: Mutex<PresentResult>,
    time_state: Mutex<TimeState>,

    pub render_targets: BTreeMap<u16, Arc<VulkanRenderTarget>>,
}

impl Drop for VulkanEntry {
    fn drop(&mut self) {
        let graphics = self
            .command_manager
            .command_entries
            .get(&CommandType::Graphics)
            .unwrap();

        graphics.wait().unwrap();

        if !self.present_result.lock().unwrap().out_of_date {
            let now = graphics.now(
                self.get_viewport()
                    .clone()
                    .as_vulkan()
                    .unwrap()
                    .sync
                    .clone(),
            );
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
                log!(
                    "No validation layers found! Vulkan SDK should be installed for proper debug. Visit https://vulkan.lunarg.com/"
                );
                return Err(CrystalError::CannotLoadLibrary);
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
            Some(vk::Extent2D {
                width: settings.width,
                height: settings.height,
            }),
        );

        let device_manager =
            DeviceManager::new(entry, instance, Some(surface.clone()), &device_extensions)?;

        log!("| picked device: [ {} ]", device_manager.device_name);
        log!(
            "| -- compression  = {}",
            device_manager.extensions.compression
        );
        log!(
            "| -- formats 4444 = {}",
            device_manager.extensions.formats_4444
        );

        let presentation =
            Presentation::new(device_manager.clone(), surface, settings.msaa_samples)?;

        let command_manager = CommandManager::new(
            device_manager.clone(),
            presentation.swapchain.swapchain_info.image_count,
        )?;

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
            device_manager: device_manager.clone(),
            command_manager,

            #[cfg(debug_assertions)]
            _debug_utils_messanger: Some(debug_utils_messanger),
            #[cfg(not(debug_assertions))]
            _debug_utils_messanger: None,

            presentation,

            present_result: Mutex::new(PresentResult::default()),
            time_state: Mutex::new(TimeState {
                timer: std::time::Instant::now(),
                delta_time: std::time::Duration::ZERO,
            }),

            render_targets,
        }))
    }

    pub fn recreate_resources(&self, width: u32, height: u32) -> CrystalResult<()> {
        self.command_manager
            .command_entries
            .get(&CommandType::Graphics)
            .clone()
            .unwrap()
            .wait()?;

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

    pub fn dispatch_compute(self: Arc<Self>, objects: &[Arc<Object>]) -> CrystalResult<()> {
        let compute = self
            .command_manager
            .command_entries
            .get(&CommandType::Compute)
            .unwrap()
            .clone();

        let sync = GpuSync::no_sync(self.device_manager.clone());

        let now = compute.now(sync.clone());

        let future = now.join(
            compute
                .record_single_time_buffer(|command_buffer, device| unsafe {
                    for object in objects {
                        let pipeline = object.pipeline.clone().as_vulkan().unwrap();
                        let groups = object.groups.unwrap();

                        device.cmd_bind_pipeline(
                            *command_buffer,
                            vk::PipelineBindPoint::COMPUTE,
                            pipeline.handle,
                        );
                        device.cmd_bind_descriptor_sets(
                            *command_buffer,
                            vk::PipelineBindPoint::COMPUTE,
                            pipeline.layout.pipeline_layout,
                            0,
                            &pipeline.layout.get_descriptor_sets(),
                            &[],
                        );

                        device.cmd_dispatch(*command_buffer, groups[0], groups[1], groups[2]);
                    }
                })
                .unwrap(),
        );

        future.flush(compute.queue.clone()).unwrap();

        Ok(())
    }

    pub fn dispatch_any(self: Arc<Self>, objects: &[Arc<Object>]) -> CrystalResult<()> {
        let graphics = self
            .command_manager
            .command_entries
            .get(&CommandType::Graphics)
            .unwrap()
            .clone();

        let compute = self
            .command_manager
            .command_entries
            .get(&CommandType::Compute)
            .unwrap()
            .clone();

        #[cfg(debug_assertions)]
        self.get_debug_data();

        let render_target_dyn = self.get_viewport();
        let render_target = render_target_dyn
            .clone()
            .as_vulkan()
            .expect("fatal: wrong type of RenderTarget, expected: VulkanRenderTarget");

        let sync = render_target.sync.clone();
        let graphics_now = graphics.now(sync.clone());

        let present_result = *self.present_result.lock().unwrap();

        if present_result.out_of_date {
            self.presentation.swapchain.recreate(None)?;
        }

        if present_result.suboptimal {
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

        sync.lock().unwrap().wait_render().unwrap();

        let mut timer = self.time_state.lock().unwrap();
        timer.delta_time = timer.timer.elapsed();
        timer.timer = std::time::Instant::now();
        drop(timer);

        match graphics_now.acquire_next_image(&self.presentation) {
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
                // return Ok(());
            }
            Err(e) => {
                log!("failed aquire next image: {:?}", e);
                return Err(CrystalError::RenderingError);
            }
            _ => (),
        };

        let compute_now = compute.now(sync.clone());

        let compute_future = compute_now.join(
            compute
                .record_command_buffer(sync.clone(), |command_buffer, device, _n_pass| unsafe {
                    for object in objects {
                        if object.groups.is_none() {
                            continue;
                        }

                        let pipeline = object.pipeline.clone().as_vulkan().unwrap();
                        let groups = object.groups.unwrap();

                        device.cmd_bind_pipeline(
                            *command_buffer,
                            vk::PipelineBindPoint::COMPUTE,
                            pipeline.handle,
                        );
                        device.cmd_bind_descriptor_sets(
                            *command_buffer,
                            vk::PipelineBindPoint::COMPUTE,
                            pipeline.layout.pipeline_layout,
                            0,
                            &pipeline.layout.get_descriptor_sets(),
                            &[],
                        );
                        device.cmd_dispatch(*command_buffer, groups[0], groups[1], groups[2]);
                    }
                })
                .unwrap(),
        );

        let graphics_future = graphics_now.join(graphics.record_command_buffer(
            sync.clone(),
            |command_buffer, device, n_pass| {
                let color = 0.5f32;
                let mut clear_color = vk::ClearColorValue::default();
                let clear_depth_stencil =
                    vk::ClearDepthStencilValue::default().depth(1.).stencil(0);
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

                let render_pass_begin = vk::RenderPassBeginInfo::default()
                    .render_pass(render_target.render_pass)
                    .framebuffer(*render_target.framebuffers[n_pass].read().unwrap())
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
                        },
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

                let mut layout_objects =
                    BTreeMap::<u64, (Arc<VulkanLayout>, Vec<Arc<Object>>)>::new();

                objects.iter().for_each(|object| {
                    if object.groups.is_none() {
                        let layout = object.pipeline.clone().as_vulkan().unwrap().layout.clone();

                        let raw = object
                            .pipeline
                            .clone()
                            .as_vulkan()
                            .unwrap()
                            .layout
                            .pipeline_layout
                            .as_raw();

                        match layout_objects.get_mut(&raw) {
                            Some((_, objects)) => {
                                objects.push(object.clone());
                            }
                            None => {
                                layout_objects.insert(raw, (layout, vec![object.clone()]));
                            }
                        }
                    }
                });

                for (_, (layout, objects)) in layout_objects {
                    layout.render(&objects, command_buffer).unwrap();
                }

                unsafe { device.cmd_end_render_pass(*command_buffer) }
            },
        )?);

        let presentation = self.presentation.clone();

        compute.queue.wait_idle().unwrap();

        compute_future.flush(compute.queue.clone()).unwrap();
        let result =
            graphics_future.swapchain_present_and_flush(graphics.queue.clone(), presentation);

        let mut result_lock = self.present_result.lock().unwrap();
        *result_lock = result;

        Ok(())
    }

    pub fn get_delta_time(&self) -> std::time::Duration {
        self.time_state.lock().unwrap().delta_time
    }

    pub fn create_layout(
        &self,
        double_buffering: bool,
        texture_num: usize,
        sampler_num: usize,
        uniform_num: usize,
        storage_num: usize,
    ) -> CrystalResult<Arc<dyn Layout>> {
        log!(
            "creating layout [ double_buffering = {} ]",
            double_buffering
        );

        Ok(layout::VulkanLayout::new(
            self.device_manager.clone(),
            texture_num,
            sampler_num,
            uniform_num,
            storage_num,
            double_buffering,
        )?)
    }

    pub fn create_sampler(
        &self,
        buffer: Arc<dyn traits::Buffer>,
        data: [u32; 3],
        anisotropy_texels: f32,
    ) -> CrystalResult<Arc<GpuSampler>> {
        log!(
            "creating texture [ width = {}, height = {} ]",
            data[0],
            data[1],
        );

        let texture = VulkanTexture::new(
            self.device_manager.clone(),
            self.command_manager.clone(),
            buffer.clone().as_vulkan().unwrap(),
            data,
            anisotropy_texels,
        )?;
        Ok(GpuSampler::from_texture(texture))
    }

    pub fn get_viewport(&self) -> Arc<dyn traits::RenderTarget> {
        self.render_targets[&0].clone()
    }

    pub fn get_debug_data(&self) -> DebugData {
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

        let debug_data = DebugData {
            used_memory: budget_props.heap_usage[0],
            aviable_memory: budget_props.heap_budget[0],
        };

        debug_data
    }

    pub fn create_buffer_mesh(&self, mesh: Arc<Mesh>) -> CrystalResult<Arc<MeshBuffer>> {
        let vertex_size = (mesh.vertices.len() * size_of::<VertexTexture>()) as u64;
        let index_size = (mesh.indices.len() * size_of::<Index>()) as u64;
        log!(
            "creating mesh [ size = {:.1} MB] ",
            (vertex_size + index_size) as f32 / 1024. / 1024.
        );

        let mut buffer_info = BufferInfo {
            size: vertex_size,
            usage: vk::BufferUsageFlags::VERTEX_BUFFER,
            properties: vk::MemoryPropertyFlags::HOST_VISIBLE
                | vk::MemoryPropertyFlags::HOST_COHERENT,
            count: 1,
        };

        let vertex_buffer_manager =
            BufferManager::new(self.device_manager.clone(), buffer_info.clone(), None)?;

        vertex_buffer_manager
            .get_memory_full()
            .copy_from_slice(mesh.vertices.as_bytes());

        buffer_info.usage = vk::BufferUsageFlags::INDEX_BUFFER;
        buffer_info.size = index_size;

        let index_buffer_manager =
            BufferManager::new(self.device_manager.clone(), buffer_info, None)?;

        index_buffer_manager
            .get_memory_full()
            .copy_from_slice(mesh.indices.as_bytes());

        Ok(Arc::new(MeshBuffer {
            mesh,
            vertices: vertex_buffer_manager,
            indices: index_buffer_manager,
        }))
    }

    pub fn create_buffer(
        &self,
        size: u64,
        uniform: bool,
        transfer: bool,
        enable_sync: bool,
    ) -> CrystalResult<Arc<dyn traits::Buffer>> {
        let mut usage = vk::BufferUsageFlags::STORAGE_BUFFER;

        if uniform {
            usage = vk::BufferUsageFlags::UNIFORM_BUFFER;
        }

        if transfer {
            usage |= vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC
        }

        let buffer_info = BufferInfo {
            size,
            usage,
            properties: vk::MemoryPropertyFlags::HOST_VISIBLE
                | vk::MemoryPropertyFlags::HOST_COHERENT,
            count: 1,
        };

        let render_target = self.get_viewport().clone().as_vulkan().unwrap();

        let buffer_manager = BufferManager::new(
            self.device_manager.clone(),
            buffer_info,
            if enable_sync {
                Some(render_target.sync.clone())
            } else {
                None
            },
        )?;

        Ok(buffer_manager)
    }
}
