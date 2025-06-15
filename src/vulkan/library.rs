use std::{collections::BTreeMap, ffi::CStr, sync::Arc, u64};

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
    GraphicsApi, GraphicsApiInitSettings,
    debug::log,
    errors::{CrystalError, CrystalResult},
    images::Image2D,
    traits::{self, Layout},
    vulkan::commands::GpuFuture,
};

pub struct VulkanEntry {
    command_manager: Arc<CommandManager>,
    device_manager: Arc<DeviceManager>,
    _debug_utils_messanger: Option<DebugUtilsMessanger>,

    presentation: Presentation,
    future: Option<Arc<GpuFuture>>,

    pub render_targets: BTreeMap<u16, Arc<VulkanRenderTarget>>,
}

impl GraphicsApi for VulkanEntry {
    fn get_current_frame(&self) -> usize {
        self.presentation.current_frame
    }

    fn render_and_present(&mut self, layouts: Vec<Arc<dyn Layout>>) -> CrystalResult<()> {
        let render_target_dyn = self.get_viewport();
        let render_target = render_target_dyn
            .clone()
            .as_vulkan()
            .expect("fatal: wrong type of RenderTarget, expected: VulkanRenderTarget");

        let now = match self.future.clone() {
            None => GpuFuture::now(self.device_manager.clone()),
            Some(future) => future.wait().unwrap(),
        };

        match now.clone().acquire_next_image(&self.presentation) {
            Ok((idx, _)) => self.presentation.image_index = idx,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                let images = self.presentation.swapchain.recreate()?;
                render_target.update_resources(self.presentation.swapchain.extent(), images)?;
                return Ok(());
            }
            Err(e) => {
                log!("failed aquire next image: {}", e);
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

        let future = now.join(self.command_manager.graphics.clone().unwrap().record_command_buffer(
                self.presentation.current_frame,
                |command_buffer, device| {
                    let render_pass_begin = vk::RenderPassBeginInfo::default()
                        .render_pass(render_target.render_pass)
                        .framebuffer(*render_target.framebuffers[self.presentation.image_index as usize].read().unwrap())
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

                    for layout in layouts.clone() {
                        let layout_downcasted = match layout.as_vulkan() {
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
            )?);

        match future
            .clone()
            .then_swapchain_present(
                self.command_manager.present.clone().unwrap(),
                &self.presentation,
            )
            .result()
        {
            Ok(()) => self.future = Some(future),
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR | vk::Result::SUBOPTIMAL_KHR) => {
                let images = self
                    .presentation
                    .swapchain
                    .recreate()
                    .expect("cannot recreate swapchain");

                render_target
                    .update_resources(self.presentation.swapchain.extent(), images)
                    .expect("cannot update render target size");
                self.future = None;
            }
            Err(e) => {
                panic!("failed to present queue: {}", e);
            }
        }

        self.presentation.current_frame =
            (self.presentation.current_frame + 1) % self.presentation.frames_in_flight as usize;

        Ok(())
    }

    fn create_layout(
        &self,
        frames_in_flight: u32,
        image_view_sampled_num: u32,
        max_instance_num: u64,
        buffers: &[(bool, u64)],
    ) -> CrystalResult<Arc<dyn Layout>> {
        Ok(layout::VulkanLayout::new(
            self.device_manager.clone(),
            image_view_sampled_num,
            frames_in_flight,
            max_instance_num,
            buffers,
        )?)
    }

    fn create_texture(
        &self,
        image: &Image2D,
        anisotropy_texels: f32,
    ) -> CrystalResult<Arc<dyn traits::Texture>> {
        let texture = VulkanTexture::new(self.device_manager.clone(), image, anisotropy_texels)?;

        texture.prepare_texture_image(&self.command_manager)?;
        Ok(texture)
    }

    fn get_viewport(&self) -> Arc<dyn traits::RenderTarget> {
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
            entry: entry.clone(),
            instance: instance.clone(),
            device: logical_device.clone(),
            physical_device,
            memory_properties,
            queue_families_indices,
            device_properties,
        });

        let command_manager =
            CommandManager::new(device_manager.clone(), settings.viewport_frames_in_flight)?;

        let presentation = Presentation::new(
            device_manager.clone(),
            surface,
            settings.viewport_frames_in_flight,
            settings.msaa_samples,
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

        Ok(Box::new(Self {
            device_manager,
            command_manager,

            #[cfg(debug_assertions)]
            _debug_utils_messanger: Some(debug_utils_messanger),
            #[cfg(not(debug_assertions))]
            _debug_utils_messanger: None,

            presentation,
            future: None,

            render_targets,
        }))
    }

    pub fn create_texture(
        &self,
        image: &Image2D,
        command_manager: &CommandManager,
        anisotropy_texels: f32,
    ) -> CrystalResult<Arc<VulkanTexture>> {
        let texture = VulkanTexture::new(self.device_manager.clone(), image, anisotropy_texels)?;
        texture.prepare_texture_image(command_manager)?;
        Ok(texture)
    }
}
