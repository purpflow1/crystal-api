use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use std::sync::Arc;
use vulkano::{
    device::DeviceOwned,
    image::{
        ImageAspects, ImageSubresourceRange,
        sampler::{ComponentMapping, ComponentSwizzle},
        view::{ImageView, ImageViewCreateInfo, ImageViewType},
    },
    memory::allocator::StandardMemoryAllocator,
    render_pass::RenderPass,
    swapchain::{Swapchain, SwapchainCreateInfo},
};

use crate::{
    debug::log,
    errors::{CrystalError, CrystalResult},
};

use super::rendering::VulkanRenderTarget;

pub struct SwapChainSupportDetails {
    pub formats: Vec<(vulkano::format::Format, vulkano::swapchain::ColorSpace)>,
    pub present_modes: Vec<vulkano::swapchain::PresentMode>,
    pub capabilities: vulkano::swapchain::SurfaceCapabilities,
}

pub struct Presentation {
    pub surface: Arc<vulkano::swapchain::Surface>,
    swapchain_create_info: Option<SwapchainCreateInfo>,
    pub swapchain: Option<Arc<Swapchain>>,
    pub image_views: Option<Vec<Arc<ImageView>>>,
}

impl Presentation {
    pub fn new<T: HasWindowHandle + HasDisplayHandle>(
        instance: Arc<vulkano::instance::Instance>,
        window: &T,
    ) -> CrystalResult<Self> {
        let surface =
            match unsafe { vulkano::swapchain::Surface::from_window_ref(instance, window) } {
                Ok(surace) => surace,
                Err(e) => {
                    log!("cannot create surface for window: {:?}", e);
                    return Err(CrystalError::PresentationError);
                }
            };

        Ok(Presentation {
            surface,
            swapchain_create_info: None,
            swapchain: None,
            image_views: None,
        })
    }

    pub fn create_swapchain_info(
        &self,
        swap_chain_support_details: &SwapChainSupportDetails,
    ) -> CrystalResult<vulkano::swapchain::SwapchainCreateInfo> {
        let swap_surface_format =
            match swap_chain_support_details
                .formats
                .iter()
                .find(|(format, color_space)| {
                    *format == vulkano::format::Format::B8G8R8A8_SRGB
                        && *color_space == vulkano::swapchain::ColorSpace::SrgbNonLinear
                }) {
                Some(&format) => format,
                None => {
                    log!("not found required swap surface format");
                    return Err(CrystalError::SwapChainIsNotSupported);
                }
            };

        let swap_extent = match swap_chain_support_details.capabilities.current_extent {
            Some(extent) => extent,
            None => {
                log!("no surface extent");
                return Err(CrystalError::SwapChainIsNotSupported);
            }
        };
        if swap_extent[0] == u32::MAX {
            log!("unknown surface extent");
            return Err(CrystalError::SwapChainIsNotSupported);
        };

        let image_count = match swap_chain_support_details.capabilities.max_image_count {
            Some(max_image_count) => max_image_count,
            None => swap_chain_support_details.capabilities.min_image_count,
        };

        let swapchain_create_info = vulkano::swapchain::SwapchainCreateInfo {
            min_image_count: image_count,
            image_format: swap_surface_format.0,
            image_color_space: swap_surface_format.1,
            image_extent: swap_extent,
            image_array_layers: 1,
            image_usage: vulkano::image::ImageUsage::COLOR_ATTACHMENT,
            pre_transform: swap_chain_support_details.capabilities.current_transform,
            composite_alpha: vulkano::swapchain::CompositeAlpha::Opaque,
            image_sharing: vulkano::sync::Sharing::Exclusive,
            present_mode: vulkano::swapchain::PresentMode::Fifo, // TODO add mailbox check
            clipped: true,
            ..Default::default()
        };

        Ok(swapchain_create_info)
    }

    fn query_swap_chain_support(
        &self,
        physical_device: Arc<vulkano::device::physical::PhysicalDevice>,
    ) -> CrystalResult<SwapChainSupportDetails> {
        let surface_info = vulkano::swapchain::SurfaceInfo::default();

        let formats = match physical_device.surface_formats(&self.surface, surface_info.clone()) {
            Ok(data) => data,
            Err(e) => {
                log!("cannot get physical device surface formats: {:?}", e);
                return Err(CrystalError::CannotInitDevice);
            }
        };

        let capabilities =
            match physical_device.surface_capabilities(&self.surface, surface_info.clone()) {
                Ok(data) => data,
                Err(e) => {
                    log!("cannot get physical device surface capabilities: {:?}", e);
                    return Err(CrystalError::SwapChainIsNotSupported);
                }
            };

        let present_modes = match physical_device.surface_present_modes(&self.surface, surface_info)
        {
            Ok(data) => data,
            Err(e) => {
                log!("cannot get physical device surface present modes: {:?}", e);
                return Err(CrystalError::SwapChainIsNotSupported);
            }
        };

        if formats.is_empty() || present_modes.is_empty() {
            return Err(CrystalError::SwapChainIsNotSupported);
        }

        Ok(SwapChainSupportDetails {
            formats,
            capabilities,
            present_modes,
        })
    }

    pub fn create_swapchain(&mut self, render_pass: Arc<RenderPass>) -> CrystalResult<()> {
        let device = render_pass.device();

        if self.swapchain_create_info.is_none() {
            let swapchain_support_details =
                self.query_swap_chain_support(device.physical_device().clone())?;
            self.swapchain_create_info =
                Some(self.create_swapchain_info(&swapchain_support_details)?);
        };

        let swapchain_create_info = self.swapchain_create_info.as_ref().unwrap();

        let mut swapchain_image_views = vec![];

        let (swapchain, images) = match Swapchain::new(
            device.clone(),
            self.surface.clone(),
            swapchain_create_info.clone(),
        ) {
            Ok(swapchain) => swapchain,
            Err(e) => {
                log!("cannot create swapchain: {:?}", e);
                return Err(CrystalError::SwapChainError);
            }
        };

        for image in images {
            let create_info = ImageViewCreateInfo {
                view_type: ImageViewType::Dim2d,
                format: swapchain_create_info.image_format,
                component_mapping: ComponentMapping {
                    r: ComponentSwizzle::Identity,
                    g: ComponentSwizzle::Identity,
                    b: ComponentSwizzle::Identity,
                    a: ComponentSwizzle::Identity,
                },
                subresource_range: ImageSubresourceRange {
                    aspects: ImageAspects::COLOR,
                    mip_levels: 0..1,
                    array_layers: 0..1,
                },
                ..Default::default()
            };

            let image_view = match ImageView::new(image.clone(), create_info) {
                Ok(image_view) => image_view,
                Err(e) => {
                    log!("cannot create image view: {:?}", e);
                    return Err(CrystalError::SwapChainError);
                }
            };

            swapchain_image_views.push(image_view);
        }

        self.swapchain = Some(swapchain);
        self.image_views = Some(swapchain_image_views);

        Ok(())
    }

    pub fn create_render_target(
        &mut self,
        memory_allocator: Arc<StandardMemoryAllocator>,
        msaa_samples: u8,
    ) -> CrystalResult<VulkanRenderTarget> {
        let device = memory_allocator.device();

        let swap_chain_support_details =
            self.query_swap_chain_support(device.physical_device().clone())?;

        let swapchain_create_info = self.create_swapchain_info(&swap_chain_support_details)?;

        let render_pass = VulkanRenderTarget::create_render_pass(
            device.clone(),
            swapchain_create_info.image_format,
            msaa_samples,
        )?;

        self.create_swapchain(render_pass.clone())?;

        let viewport_render_target = VulkanRenderTarget::new(
            render_pass,
            memory_allocator,
            self.image_views.as_ref().unwrap(),
            swapchain_create_info.image_extent,
        )?;

        self.swapchain_create_info = Some(swapchain_create_info);

        Ok(viewport_render_target)
    }
}
