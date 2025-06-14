use std::{cell::RefCell, collections::BTreeMap, ffi::CStr, sync::Arc, u64};

use ash::vk::{self, EXT_DEBUG_UTILS_NAME, KHR_SWAPCHAIN_NAME};
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
    GraphicsApi, GraphicsApiInitSettings, RenderTarget,
    debug::log,
    errors::{CrystalError, CrystalResult},
    images::Image2D,
    traits::{self, Layout},
};

pub struct VulkanEntry {
    entry: ash::Entry,
    instance: ash::Instance,
    command_manager: CommandManager,

    device_manager: Arc<DeviceManager>,
    _debug_utils_messanger: Option<DebugUtilsMessanger>,

    presentation: Presentation,

    pub render_targets: BTreeMap<u16, Arc<RefCell<VulkanRenderTarget>>>,
}

impl Drop for VulkanEntry {
    fn drop(&mut self) {
        unsafe {
            self.instance.destroy_instance(None);
        }
    }
}

impl GraphicsApi for VulkanEntry {
    fn get_current_frame(&self) -> usize {
        self.presentation.current_frame
    }

    fn render(&mut self, layouts: &[Arc<RefCell<dyn Layout>>]) -> CrystalResult<()> {
        for (_, target) in &self.render_targets {
            let mut render_target = target.borrow_mut();
            match unsafe {
                self.device_manager.device.clone().wait_for_fences(
                    &[self.presentation.in_flight_fences[self.presentation.current_frame]],
                    true,
                    u64::MAX,
                )
            } {
                Ok(()) => {}
                Err(e) => {
                    log!("failed waiting for fences: {}", e);
                    return Err(CrystalError::RenderingError);
                }
            };

            let command_entry = match &self.command_manager.graphics {
                Some(command_entry) => command_entry,
                None => {
                    log!("no graphics command entry");
                    return Err(CrystalError::CommandManagerError);
                }
            };

            let image_index;

            match unsafe {
                self.presentation.swapchain.acquire_next_image(
                    self.presentation.swapchain_khr,
                    u64::MAX,
                    self.presentation.image_available_semaphores[self.presentation.current_frame],
                    vk::Fence::null(),
                )
            } {
                Ok((idx, _)) => image_index = idx,
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    self.presentation.recreate_swapchain()?;
                    render_target.update_size(
                        self.presentation.extent.width,
                        self.presentation.extent.height,
                    )?;
                    continue;
                }
                Err(e) => {
                    log!("failed aquire next image: {}", e);
                    return Err(CrystalError::RenderingError);
                }
            };

            match unsafe {
                self.device_manager.device.clone().reset_fences(&[self
                    .presentation
                    .in_flight_fences[self.presentation.current_frame]])
            } {
                Ok(()) => {}
                Err(e) => {
                    log!("failed reset fences: {}", e);
                    return Err(CrystalError::RenderingError);
                }
            };

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

            command_entry.reset_command_buffer(self.presentation.current_frame)?;
            command_entry.record_command_buffer(
                self.presentation.current_frame,
                |command_buffer, device| {
                    let render_pass_begin = vk::RenderPassBeginInfo::default()
                        .render_pass(render_target.render_pass)
                        .framebuffer(render_target.framebuffers[image_index as usize])
                        .render_area(vk::Rect2D {
                            offset: vk::Offset2D::default().x(0).y(0),
                            extent: vk::Extent2D {
                                width: render_target
                                    .extent
                                    .width
                                    .min(self.presentation.extent.width),
                                height: render_target
                                    .extent
                                    .height
                                    .min(self.presentation.extent.height),
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
                        .width(render_target.extent.width as f32)
                        .height(render_target.extent.height as f32)
                        .max_depth(1.);
                    let viewports = &[viewport];
                    unsafe { device.cmd_set_viewport(*command_buffer, 0, viewports) }
                    let scissor = vk::Rect2D::default().extent(render_target.extent);
                    let scissors = &[scissor];
                    unsafe { device.cmd_set_scissor(*command_buffer, 0, scissors) }

                    for layout in layouts {
                        let mut layout_borrowed = layout.borrow_mut();

                        let layout_downcasted = match layout_borrowed.as_vulkan_mut() {
                            Some(layout) => layout,
                            None => {
                                panic!(
                                    "fatal: wrong layout type passed into render, expected vulkan"
                                )
                            }
                        };
                        layout_downcasted
                            .render(
                                self.device_manager.clone(),
                                command_buffer,
                                self.presentation.frames_in_flight as usize,
                                self.presentation.current_frame,
                            )
                            .unwrap();
                    }

                    unsafe { device.cmd_end_render_pass(*command_buffer) }
                },
            )?;

            let queue = unsafe {
                self.device_manager
                    .device
                    .get_device_queue(self.presentation.queue_family_indices[1], 0)
            };

            let swapchains = &[self.presentation.swapchain_khr];
            let image_indices = &[image_index];

            self.command_manager
                .graphics
                .as_ref()
                .unwrap()
                .submit_command_buffer(
                    self.presentation.current_frame,
                    &[self.presentation.image_available_semaphores
                        [self.presentation.current_frame]],
                    &[self.presentation.render_finished_semaphores
                        [self.presentation.current_frame]],
                    &[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT],
                    self.presentation.in_flight_fences[self.presentation.current_frame],
                )?;

            let wait_semaphores =
                &[self.presentation.render_finished_semaphores[self.presentation.current_frame]];

            let present_info = vk::PresentInfoKHR::default()
                .wait_semaphores(wait_semaphores)
                .swapchains(swapchains)
                .image_indices(image_indices);
            match unsafe {
                self.presentation
                    .swapchain
                    .queue_present(queue, &present_info)
            } {
                Ok(_) => (),
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR | vk::Result::SUBOPTIMAL_KHR) => {
                    self.presentation
                        .recreate_swapchain()
                        .expect("cannot recreate swapchain");
                    render_target
                        .update_size(
                            self.presentation.extent.width,
                            self.presentation.extent.height,
                        )
                        .expect("cannot update render target size");
                }
                Err(e) => {
                    panic!("failed to present queue: {}", e);
                }
            }

            self.presentation.current_frame =
                (self.presentation.current_frame + 1) % self.presentation.frames_in_flight as usize;
        }

        Ok(())
    }

    fn create_layout(
        &self,
        frames_in_flight: u32,
        image_view_sampled_num: u32,
        max_instance_num: u64,
        buffers: &[(bool, u64)],
    ) -> CrystalResult<Arc<RefCell<dyn Layout>>> {
        Ok(Arc::new(RefCell::new(layout::VulkanLayout::new(
            self.device_manager.clone(),
            image_view_sampled_num,
            frames_in_flight,
            max_instance_num,
            buffers,
        )?)))
    }

    fn create_texture(
        &self,
        image: &Image2D,
        anisotropy_texels: f32,
    ) -> CrystalResult<Arc<dyn traits::Texture>> {
        let mut texture = VulkanTexture::new(
            &self.instance,
            self.device_manager.clone(),
            image,
            anisotropy_texels,
        )?;

        texture.prepare_texture_image(&self.command_manager)?;
        Ok(Arc::new(texture))
    }

    fn get_viewport(&self) -> Arc<RefCell<dyn traits::RenderTarget>> {
        self.render_targets[&0].clone()
    }
}

impl VulkanEntry {
    pub fn with_presentation<T: HasWindowHandle + HasDisplayHandle>(
        settings: &GraphicsApiInitSettings,
        window: &T,
    ) -> CrystalResult<Box<dyn GraphicsApi>> {
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
            Ok(entry) => entry,
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
            Ok(instance) => instance,
        };

        #[cfg(debug_assertions)]
        let debug_utils_messanger = {
            log!("creating debug utils");
            create_debug_utils_messanger(&entry, &instance)?
        };

        if settings.viewport_frames_in_flight == 0 {
            panic!(
                "fatal: wrong frames in flight count: {}",
                settings.viewport_frames_in_flight
            );
        }

        let surface = Presentation::create_surface(&entry, &instance, window);

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
            device: logical_device.clone(),
            physical_device,
            memory_properties,
            queue_families_indices,
            device_properties,
        });

        let command_manager = CommandManager::new(
            device_manager.device.clone(),
            &device_manager.queue_families_indices,
            settings.viewport_frames_in_flight,
        )?;

        let presentation = Presentation::new(
            &instance,
            device_manager.clone(),
            surface,
            settings.viewport_frames_in_flight,
            settings.msaa_samples,
        )?;

        let viewport_render_target = VulkanRenderTarget::new(
            &instance,
            device_manager.clone(),
            presentation.image_format,
            vk::Extent2D {
                width: settings.width,
                height: settings.height,
            },
            &presentation.swapchain_image_views,
            presentation.msaa_samples,
        )?;

        let mut render_targets = BTreeMap::new();

        render_targets.insert(0, Arc::new(RefCell::new(viewport_render_target)));

        Ok(Box::new(Self {
            entry,
            instance,
            device_manager,
            command_manager,
            #[cfg(debug_assertions)]
            _debug_utils_messanger: Some(debug_utils_messanger),
            #[cfg(not(debug_assertions))]
            _debug_utils_messanger: None,

            presentation,

            render_targets,
        }))
    }

    pub fn create_texture(
        &self,
        image: &Image2D,
        command_manager: &CommandManager,
        anisotropy_texels: f32,
    ) -> CrystalResult<VulkanTexture> {
        let mut texture = VulkanTexture::new(
            &self.instance,
            self.device_manager.clone(),
            image,
            anisotropy_texels,
        )?;
        texture.prepare_texture_image(command_manager)?;
        Ok(texture)
    }
}
