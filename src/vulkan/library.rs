use std::{cell::RefCell, collections::BTreeMap, sync::Arc, u64};

use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use vulkano::{
    Validated, VulkanError,
    command_buffer::{RenderPassBeginInfo, SubpassBeginInfo, SubpassContents, SubpassEndInfo},
    device::DeviceOwned,
    format::ClearValue,
    memory::allocator::StandardMemoryAllocator,
    swapchain::{self, SwapchainPresentInfo},
    sync::{self, GpuFuture, future::FenceSignalFuture},
};

use super::{
    commands::CommandManager, devices::create_logical_device, images::VulkanTexture, layout,
    presentation::Presentation, rendering::VulkanRenderTarget,
};

#[cfg(debug_assertions)]
use crate::vulkan::debug_callback::create_debug_utils_messanger_create_info;
use crate::{
    GraphicsApi, GraphicsApiInitSettings, RenderTarget,
    debug::log,
    errors::{CrystalError, CrystalResult},
    images::Image2D,
    traits::{self, Layout},
    vulkan::devices::pick_physical_device,
};

pub struct VulkanEntry {
    current_future: Option<Arc<FenceSignalFuture<Box<dyn GpuFuture>>>>,
    memory_allocator: Arc<StandardMemoryAllocator>,

    command_manager: CommandManager,
    presentation: Presentation,

    pub render_targets: BTreeMap<u16, Arc<RefCell<VulkanRenderTarget>>>,
}

impl GraphicsApi for VulkanEntry {
    fn render(
        &mut self,
        layouts: &[Arc<RefCell<dyn Layout>>],
        render_target: Arc<RefCell<dyn RenderTarget>>,
    ) -> CrystalResult<()> {
        let mut render_target_borrowed = render_target.borrow_mut();

        let render_target_downcasted = match render_target_borrowed.as_vulkan_mut() {
            Some(render_target) => render_target,
            None => {
                panic!("fatal: wrong render target type passed into render, expected vulkan")
            }
        };

        let graphics_entry = self.command_manager.graphics.as_ref().unwrap();

        let future = if let Some(future) = self.current_future.clone() {
            match future.wait(None) {
                Ok(()) => (),
                Err(e) => {
                    log!("cannot wait for fence: {:?}", e);
                    return Err(CrystalError::SyncError);
                }
            };

            future.boxed()
        } else {
            let mut now = sync::now(self.memory_allocator.device().clone());
            now.cleanup_finished();

            now.boxed()
        };

        let (image_index, suboptimal, acquire_future) = match swapchain::acquire_next_image(
            self.presentation.swapchain.as_ref().unwrap().clone(),
            None,
        )
        .map_err(Validated::unwrap)
        {
            Ok(r) => r,
            Err(VulkanError::OutOfDate) => {
                self.presentation
                    .create_swapchain(render_target_downcasted.render_pass.clone())?;
                return Ok(());
            }
            Err(e) => panic!("failed to acquire next image: {e}"),
        };

        if suboptimal {
            self.presentation
                .create_swapchain(render_target_downcasted.render_pass.clone())?;
        }

        let color = 0.2f32;
        let clear_value_color = ClearValue::Float([color; 4]);
        let clear_value_stencil = ClearValue::DepthStencil((1., 0));
        let clear_values = vec![Some(clear_value_color), Some(clear_value_stencil)];

        let command_buffer = graphics_entry.record_command_buffer(|command_buffer_builder| {
            let mut render_pass_begin_info = RenderPassBeginInfo::framebuffer(
                render_target_downcasted.framebuffers[render_target_downcasted.current_frame]
                    .clone(),
            );
            render_pass_begin_info.clear_values = clear_values.clone();

            let subpass_begin_info = SubpassBeginInfo {
                contents: SubpassContents::Inline,
                ..Default::default()
            };

            command_buffer_builder
                .begin_render_pass(render_pass_begin_info, subpass_begin_info)
                .expect("fatal: cannot begin rendering");

            for layout in layouts {
                let mut layout_borrowed = layout.borrow_mut();

                let layout_downcasted = match layout_borrowed.as_vulkan_mut() {
                    Some(layout) => layout,
                    None => {
                        panic!("fatal: wrong layout type passed into render, expected vulkan")
                    }
                };
                layout_downcasted
                    .render(command_buffer_builder, render_target_downcasted)
                    .unwrap();
            }

            command_buffer_builder
                .end_render_pass(SubpassEndInfo::default())
                .expect("fatal: connot end rendering");
        })?;

        let queue = self
            .command_manager
            .graphics
            .as_ref()
            .unwrap()
            .queue
            .clone();

        let future = future
            .join(acquire_future)
            .then_execute(queue.clone(), command_buffer)
            .unwrap()
            .then_swapchain_present(
                queue.clone(),
                SwapchainPresentInfo::swapchain_image_index(
                    self.presentation.swapchain.as_ref().unwrap().clone(),
                    image_index,
                ),
            )
            .boxed()
            .then_signal_fence_and_flush()
            .unwrap();

        self.current_future = Some(Arc::new(future));

        render_target_downcasted.current_frame = (render_target_downcasted.current_frame + 1)
            % render_target_downcasted.frames_in_flight;

        Ok(())
    }

    fn create_layout_from_data(
        &self,
        frames_in_flight: u32,
        image_view_sampled_num: u32,
        max_instance_num: u64,
        buffers: &[(bool, u64)],
    ) -> CrystalResult<Arc<RefCell<dyn Layout>>> {
        Ok(Arc::new(RefCell::new(layout::VulkanLayout::new_static(
            self.memory_allocator.clone(),
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
        let mut texture =
            VulkanTexture::new(self.memory_allocator.clone(), image, anisotropy_texels)?;

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
    ) -> CrystalResult<Arc<RefCell<dyn GraphicsApi>>> {
        assert!(
            settings.viewport_frames_in_flight > 0,
            "fatal: wrong frames in flight count: {}",
            settings.viewport_frames_in_flight
        );

        let layers = vec![
            #[cfg(debug_assertions)]
            "VK_LAYER_KHRONOS_validation".to_string(),
        ];

        let device_extensions = vulkano::device::DeviceExtensions {
            khr_swapchain: true,
            ..vulkano::device::DeviceExtensions::empty()
        };

        let device_features = vulkano::device::DeviceFeatures {
            sampler_anisotropy: true,
            ..Default::default()
        };

        let extensions = match vulkano::swapchain::Surface::required_extensions(window) {
            Ok(extensions) => extensions,
            Err(e) => {
                log!("cannot get required surface extensions: {:?}", e);
                return CrystalResult::Err(CrystalError::CannotLoadLibrary);
            }
        };

        let entry = match vulkano::VulkanLibrary::new() {
            Ok(entry) => entry,
            Err(e) => {
                log!("cannot load vulkan entry: {:?}", e);
                return CrystalResult::Err(CrystalError::CannotLoadLibrary);
            }
        };

        #[cfg(debug_assertions)]
        match entry.supported_extensions_with_layers(layers.iter().map(|x| x.as_str())) {
            Ok(extensions) => {
                if !extensions.ext_debug_utils {
                    panic!("validation is not supported");
                }
            }
            Err(e) => {
                log!("cannot get supported extensions with layers: {:?}", e);
                return CrystalResult::Err(CrystalError::CannotLoadLibrary);
            }
        };

        let create_info = vulkano::instance::InstanceCreateInfo {
            enabled_extensions: vulkano::instance::InstanceExtensions {
                #[cfg(debug_assertions)]
                ext_debug_utils: true,
                ..extensions
            },
            debug_utils_messengers: vec![
                #[cfg(debug_assertions)]
                create_debug_utils_messanger_create_info(),
            ],
            enabled_layers: layers,
            ..Default::default()
        };

        let instance = match vulkano::instance::Instance::new(entry, create_info) {
            Ok(instance) => instance,
            Err(e) => {
                log!("cannot create vulkan instance: {:?}", e);
                return CrystalResult::Err(CrystalError::ConnotInitLibrary);
            }
        };

        log!("Vulkan API version: {}", instance.api_version());

        let mut presentation = Presentation::new(instance.clone(), window)?;

        let physical_device = pick_physical_device(instance.clone(), &device_extensions)?;

        let (device, queues) = create_logical_device(
            physical_device.clone(),
            Some(&presentation),
            &device_extensions,
            &device_features,
        )?;

        let command_manager = CommandManager::new(&queues)?;

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

        let viewport_render_target =
            presentation.create_render_target(memory_allocator.clone(), settings.msaa_samples)?;

        let mut render_targets = BTreeMap::new();

        render_targets.insert(0, Arc::new(RefCell::new(viewport_render_target)));

        Ok(Arc::new(RefCell::new(Self {
            current_future: None,
            memory_allocator,

            command_manager,

            presentation,

            render_targets,
        })))
    }

    pub fn create_texture(
        &self,
        image: &Image2D,
        command_manager: &CommandManager,
        anisotropy_texels: f32,
    ) -> CrystalResult<VulkanTexture> {
        let mut texture =
            VulkanTexture::new(self.memory_allocator.clone(), image, anisotropy_texels)?;
        texture.prepare_texture_image(command_manager)?;
        Ok(texture)
    }
}
