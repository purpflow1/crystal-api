use std::sync::Arc;

use ash::{
    Entry,
    vk::{self, PresentModeKHR, SurfaceCapabilitiesKHR, SurfaceFormatKHR},
};
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};

use crate::{
    debug::log,
    errors::{CrystalError, CrystalResult},
};

use super::{devices::DeviceManager, rendering::VulkanRenderTarget};

pub struct SwapChainSupportDetails {
    pub formats: Vec<SurfaceFormatKHR>,
    pub present_modes: Vec<PresentModeKHR>,
    pub capabilities: SurfaceCapabilitiesKHR,
}

pub struct Presentation {
    pub surface: ash::khr::surface::Instance,
    pub surface_khr: vk::SurfaceKHR,
    frames_in_flight: u32,
    pub msaa_samples: u8,
}

impl Presentation {
    pub fn new(
        entry: &Entry,
        instance: &ash::Instance,
        handles: (RawDisplayHandle, RawWindowHandle),
        frames_in_flight: u32,
        msaa_samples: u8,
    ) -> CrystalResult<Self> {
        let surface = ash::khr::surface::Instance::new(entry, instance);
        let surface_khr = unsafe {
            ash_window::create_surface(&entry, &instance, handles.0, handles.1, None).unwrap()
        };

        Ok(Presentation {
            surface,
            surface_khr,
            frames_in_flight,
            msaa_samples,
        })
    }

    fn query_swap_chain_support(
        &self,
        physical_device: &vk::PhysicalDevice,
    ) -> CrystalResult<SwapChainSupportDetails> {
        let formats = match unsafe {
            self.surface
                .get_physical_device_surface_formats(*physical_device, self.surface_khr)
        } {
            Ok(data) => data,
            Err(e) => {
                log!("cannot get physical device surface formats: {}", e);
                return Err(CrystalError::CannotInitDevice);
            }
        };

        let capabilities = match unsafe {
            self.surface
                .get_physical_device_surface_capabilities(*physical_device, self.surface_khr)
        } {
            Ok(data) => data,
            Err(e) => {
                log!("cannot get physical device surface capabilities: {}", e);
                return Err(CrystalError::CannotInitDevice);
            }
        };

        let present_modes = match unsafe {
            self.surface
                .get_physical_device_surface_present_modes(*physical_device, self.surface_khr)
        } {
            Ok(data) => data,
            Err(e) => {
                log!("cannot get physical device surface present modes: {}", e);
                return Err(CrystalError::CannotInitDevice);
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

    pub fn create_swapchain_info(
        &self,
        device_manager: Arc<DeviceManager>,
    ) -> CrystalResult<vk::SwapchainCreateInfoKHR> {
        let swap_chain_support_details =
            self.query_swap_chain_support(&device_manager.physical_device)?;

        let swap_surface_format = match swap_chain_support_details.formats.iter().find(|format| {
            format.format == vk::Format::B8G8R8A8_SRGB
                && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        }) {
            Some(&format) => format,
            None => {
                log!("not found required swap surface format");
                return Err(CrystalError::SwapChainIsNotSupported);
            }
        };

        let swap_extent =
            if swap_chain_support_details.capabilities.current_extent.width != u32::MAX {
                swap_chain_support_details.capabilities.current_extent
            } else {
                log!("unknown surface extent");
                return Err(CrystalError::SwapChainIsNotSupported);
            };

        let image_count = {
            let max_image_count = swap_chain_support_details.capabilities.max_image_count;
            let min_image_count = swap_chain_support_details.capabilities.min_image_count;
            if max_image_count > 0 {
                max_image_count
            } else {
                min_image_count
            }
        };

        let swap_present_mode = match swap_chain_support_details
            .present_modes
            .iter()
            .find(|&&mode| mode == vk::PresentModeKHR::MAILBOX)
        {
            Some(&mode) => mode,
            None => vk::PresentModeKHR::FIFO,
        };

        let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(self.surface_khr)
            .min_image_count(image_count)
            .image_format(swap_surface_format.format)
            .image_color_space(swap_surface_format.color_space)
            .image_extent(swap_extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .pre_transform(swap_chain_support_details.capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .present_mode(swap_present_mode)
            .clipped(true)
            .old_swapchain(vk::SwapchainKHR::null());

        Ok(swapchain_create_info)
    }

    pub fn init_viewport_render_target(
        &self,
        instance: &ash::Instance,
        device_manager: Arc<DeviceManager>,
    ) -> CrystalResult<VulkanRenderTarget> {
        let mut swapchain_create_info = self.create_swapchain_info(device_manager.clone())?;

        let queue_family_indices = [
            device_manager
                .queue_families_indices
                .graphics_index
                .unwrap(),
            device_manager.queue_families_indices.present_index.unwrap(),
        ];

        if queue_family_indices[0] != queue_family_indices[1] {
            swapchain_create_info = swapchain_create_info
                .image_sharing_mode(vk::SharingMode::CONCURRENT)
                .queue_family_indices(&queue_family_indices)
        }

        let viewport_render_target = VulkanRenderTarget::new(
            instance,
            device_manager,
            swapchain_create_info,
            self.frames_in_flight,
            self.msaa_samples,
        )?;

        Ok(viewport_render_target)
    }
}
