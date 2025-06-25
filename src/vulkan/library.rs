use std::{
    collections::{BTreeMap, HashSet},
    ffi::CStr,
    sync::{Arc, Mutex},
    thread::JoinHandle,
};

use ash::vk::{self, EXT_DEBUG_UTILS_NAME, Handle, KHR_SWAPCHAIN_NAME};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

use super::{
    commands::CommandManager,
    debug_callback::{DebugUtilsMessanger, create_debug_utils_messanger},
    devices::{DeviceManager, create_logical_device, pick_physical_device},
    images::VulkanTexture,
    layout,
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
        let mut handle_lock = self.thread_handle.lock().unwrap();

        let mut swapchain_out_of_date = false;

        let now = match &*handle_lock {
            Some(_handle) => match handle_lock.take().unwrap().join().unwrap() {
                Ok(n) => n.wait().unwrap(),
                Err((vk::Result::ERROR_OUT_OF_DATE_KHR, sync)) => {
                    swapchain_out_of_date = true;
                    GpuFuture::now_with_sync(self.device_manager.clone(), sync)
                }
                Err((vk::Result::SUBOPTIMAL_KHR, sync)) => {
                    GpuFuture::now_with_sync(self.device_manager.clone(), sync)
                }
                Err((e, _)) => {
                    panic!("failed to present queue: {}", e);
                }
            },
            None => GpuFuture::now(self.device_manager.clone()),
        };

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
            EXT_DEBUG_UTILS_NAME.as_ptr(),
        ];

        let device_extensions = [KHR_SWAPCHAIN_NAME.as_ptr()];

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

        let (physical_device, queue_families_indices) =
            pick_physical_device(&instance, Some(surface.clone()), &device_extensions)?;

        let logical_device = Arc::new(create_logical_device(
            &instance,
            physical_device,
            &queue_families_indices,
            &device_extensions,
            vk::PhysicalDeviceFeatures::default().sampler_anisotropy(true),
        )?);

        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };

        let device_properties = unsafe { instance.get_physical_device_properties(physical_device) };

        let device_manager = Arc::new(DeviceManager {
            entry: entry.clone(),
            instance: instance.clone(),
            device: logical_device.clone(),
            physical_device,
            memory_properties,
            queue_families_indices,
            device_properties,
        });

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
                unsafe {
                    self.device_manager
                        .device
                        .queue_wait_idle(self.command_manager.present.clone().unwrap().queue)
                }
                .unwrap();
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

    pub fn render_and_present(self: Arc<Self>, objects: &[Arc<Object>]) -> CrystalResult<()> {
        let render_target_dyn = self.get_viewport();
        let render_target = render_target_dyn
            .clone()
            .as_vulkan()
            .expect("fatal: wrong type of RenderTarget, expected: VulkanRenderTarget");

        let mut swapchain_out_of_date = false;
        let mut suboptimal = false;

        let mut handle_lock = self.thread_handle.lock().unwrap();
        let now = match &*handle_lock {
            Some(_handle) => match handle_lock.take().unwrap().join().unwrap() {
                Ok(n) => n.wait().unwrap(),
                Err((vk::Result::ERROR_OUT_OF_DATE_KHR, sync)) => {
                    swapchain_out_of_date = true;
                    suboptimal = true;
                    GpuFuture::now_with_sync(self.device_manager.clone(), sync)
                }
                Err((vk::Result::SUBOPTIMAL_KHR, sync)) => {
                    suboptimal = true;
                    GpuFuture::now_with_sync(self.device_manager.clone(), sync)
                }
                Err((e, _)) => {
                    panic!("failed to present queue: {}", e);
                }
            },
            None => GpuFuture::now(self.device_manager.clone()),
        };

        drop(handle_lock);

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

        let gpu_future = now.join(
            self.command_manager
                .graphics
                .clone()
                .unwrap()
                .record_command_buffer(n_pass, |command_buffer, device| {
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
                })?,
        );

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
                .graphics
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
}
