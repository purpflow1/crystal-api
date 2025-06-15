use std::sync::{Arc, Mutex, RwLock};

use ash::{
    Entry,
    vk::{self, PresentModeKHR, SurfaceCapabilitiesKHR, SurfaceFormatKHR},
};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

use crate::{
    debug::log,
    errors::{CrystalError, CrystalResult},
};

use super::devices::DeviceManager;

pub struct SwapChainSupportDetails {
    pub formats: Vec<SurfaceFormatKHR>,
    pub present_modes: Vec<PresentModeKHR>,
    pub capabilities: SurfaceCapabilitiesKHR,
}

pub struct PresentSurface {
    pub surface: ash::khr::surface::Instance,
    pub surface_khr: vk::SurfaceKHR,
}

impl Drop for PresentSurface {
    fn drop(&mut self) {
        unsafe {
            self.surface.destroy_surface(self.surface_khr, None);
        }
    }
}

pub struct SwapchainInfo {
    device_manager: Arc<DeviceManager>,
    surface: Arc<PresentSurface>,

    queue_family_indices: [u32; 2],
    pub surface_format: vk::SurfaceFormatKHR,
    extent: RwLock<vk::Extent2D>,
    present_mode: vk::PresentModeKHR,
    support_details: SwapChainSupportDetails,

    image_count: u32,
}

impl SwapchainInfo {
    fn new(
        device_manager: Arc<DeviceManager>,
        surface: Arc<PresentSurface>,
    ) -> CrystalResult<Arc<Self>> {
        let queue_family_indices = [
            device_manager
                .queue_families_indices
                .graphics_index
                .unwrap(),
            device_manager.queue_families_indices.present_index.unwrap(),
        ];

        let swap_chain_support_details =
            Self::query_swap_chain_support(surface.clone(), &device_manager.physical_device)?;

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

        Ok(Arc::new(Self {
            device_manager,
            surface,

            queue_family_indices,
            surface_format: swap_surface_format,
            extent: RwLock::new(swap_extent),
            present_mode: swap_present_mode,
            support_details: swap_chain_support_details,

            image_count,
        }))
    }

    fn query_swap_chain_support(
        surface: Arc<PresentSurface>,
        physical_device: &vk::PhysicalDevice,
    ) -> CrystalResult<SwapChainSupportDetails> {
        let formats = match unsafe {
            surface
                .surface
                .get_physical_device_surface_formats(*physical_device, surface.surface_khr)
        } {
            Ok(data) => data,
            Err(e) => {
                log!("cannot get physical device surface formats: {}", e);
                return Err(CrystalError::CannotInitDevice);
            }
        };

        let capabilities = match unsafe {
            surface
                .surface
                .get_physical_device_surface_capabilities(*physical_device, surface.surface_khr)
        } {
            Ok(data) => data,
            Err(e) => {
                log!("cannot get physical device surface capabilities: {}", e);
                return Err(CrystalError::CannotInitDevice);
            }
        };

        let present_modes = match unsafe {
            surface
                .surface
                .get_physical_device_surface_present_modes(*physical_device, surface.surface_khr)
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

    pub fn update_extent(&self) -> CrystalResult<()> {
        let capabilities = match unsafe {
            self.surface
                .surface
                .get_physical_device_surface_capabilities(
                    self.device_manager.physical_device,
                    self.surface.surface_khr,
                )
        } {
            Ok(data) => data,
            Err(e) => {
                log!("cannot get physical device surface capabilities: {}", e);
                return Err(CrystalError::CannotInitDevice);
            }
        };

        *self.extent.write().unwrap() = capabilities.current_extent;

        Ok(())
    }

    pub fn as_vk<'a>(&'a self) -> vk::SwapchainCreateInfoKHR<'a> {
        let mut swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(self.surface.surface_khr)
            .min_image_count(self.image_count)
            .image_format(self.surface_format.format)
            .image_color_space(self.surface_format.color_space)
            .image_extent(*self.extent.read().unwrap())
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .pre_transform(self.support_details.capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .present_mode(self.present_mode)
            .clipped(true)
            .old_swapchain(vk::SwapchainKHR::null());

        if self.queue_family_indices[0] != self.queue_family_indices[1] {
            swapchain_create_info = swapchain_create_info
                .image_sharing_mode(vk::SharingMode::CONCURRENT)
                .queue_family_indices(&self.queue_family_indices)
        }

        swapchain_create_info
    }
}

pub struct Swapchain {
    device_manager: Arc<DeviceManager>,
    pub swapchain_info: Arc<SwapchainInfo>,

    pub swapchain: RwLock<ash::khr::swapchain::Device>,
    pub swapchain_khr: RwLock<vk::SwapchainKHR>,
    pub swapchain_image_views: RwLock<Vec<vk::ImageView>>,
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        self.destroy();
    }
}

impl Swapchain {
    fn destroy(&self) {
        unsafe {
            self.device_manager.device.device_wait_idle().unwrap();

            self.swapchain_image_views
                .read()
                .unwrap()
                .iter()
                .for_each(|&image_view| {
                    self.device_manager
                        .device
                        .destroy_image_view(image_view, None)
                });
            self.swapchain
                .read()
                .unwrap()
                .destroy_swapchain(*self.swapchain_khr.read().unwrap(), None);
        }
    }

    pub fn extent(&self) -> vk::Extent2D {
        *self.swapchain_info.extent.read().unwrap()
    }

    fn from_info(
        device_manager: Arc<DeviceManager>,
        swapchain_create_info: Arc<SwapchainInfo>,
    ) -> CrystalResult<(
        ash::khr::swapchain::Device,
        vk::SwapchainKHR,
        Vec<vk::ImageView>,
    )> {
        let swapchain =
            ash::khr::swapchain::Device::new(&device_manager.instance, &device_manager.device);
        let swapchain_khr =
            match unsafe { swapchain.create_swapchain(&swapchain_create_info.as_vk(), None) } {
                Ok(swapchain_khr) => swapchain_khr,
                Err(e) => {
                    log!("cannot create swapchain: {}", e);
                    return Err(CrystalError::SwapChainError);
                }
            };

        let swapchain_images = match unsafe { swapchain.get_swapchain_images(swapchain_khr) } {
            Ok(images) => images,
            Err(e) => {
                log!("cannot get swapchain images: {}", e);
                return Err(CrystalError::SwapChainError);
            }
        };

        let mut swapchain_image_views = vec![];

        for &swapchain_image in &swapchain_images {
            let create_info = vk::ImageViewCreateInfo::default()
                .image(swapchain_image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(swapchain_create_info.surface_format.format)
                .components(vk::ComponentMapping {
                    r: vk::ComponentSwizzle::IDENTITY,
                    g: vk::ComponentSwizzle::IDENTITY,
                    b: vk::ComponentSwizzle::IDENTITY,
                    a: vk::ComponentSwizzle::IDENTITY,
                })
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(0)
                        .level_count(1)
                        .base_array_layer(0)
                        .layer_count(1),
                );

            let image_view =
                match unsafe { device_manager.device.create_image_view(&create_info, None) } {
                    Ok(image_view) => image_view,
                    Err(e) => {
                        log!("cannot create image view: {}", e);
                        return Err(CrystalError::SwapChainError);
                    }
                };

            swapchain_image_views.push(image_view);
        }

        Ok((swapchain, swapchain_khr, swapchain_image_views))
    }

    fn new(
        device_manager: Arc<DeviceManager>,
        surface: Arc<PresentSurface>,
    ) -> CrystalResult<Arc<Self>> {
        let swapchain_create_info = SwapchainInfo::new(device_manager.clone(), surface.clone())?;
        let (swapchain, swapchain_khr, swapchain_image_views) =
            Self::from_info(device_manager.clone(), swapchain_create_info.clone())?;

        Ok(Arc::new(Self {
            device_manager,
            swapchain_info: swapchain_create_info,
            swapchain: RwLock::new(swapchain),
            swapchain_khr: RwLock::new(swapchain_khr),
            swapchain_image_views: RwLock::new(swapchain_image_views),
        }))
    }

    pub fn recreate(&self) -> CrystalResult<Vec<vk::ImageView>> {
        self.destroy();
        self.swapchain_info.update_extent()?;
        let (swapchain, swapchain_khr, swapchain_image_views) =
            Swapchain::from_info(self.device_manager.clone(), self.swapchain_info.clone())?;

        *self.swapchain.write().unwrap() = swapchain;
        *self.swapchain_khr.write().unwrap() = swapchain_khr;
        *self.swapchain_image_views.write().unwrap() = swapchain_image_views.clone();

        Ok(swapchain_image_views)
    }
}

pub struct Presentation {
    device_manager: Arc<DeviceManager>,
    pub surface: Arc<PresentSurface>,
    pub swapchain: Arc<Swapchain>,

    pub current_frame: Mutex<usize>,
    pub image_index: Mutex<u32>,

    pub frames_in_flight: u32,
    pub msaa_samples: u8,
}

impl Presentation {
    pub fn create_surface<T: HasWindowHandle + HasDisplayHandle>(
        entry: &Entry,
        instance: &ash::Instance,
        window: &T,
    ) -> Arc<PresentSurface> {
        let surface = ash::khr::surface::Instance::new(entry, instance);
        let surface_khr = unsafe {
            ash_window::create_surface(
                &entry,
                &instance,
                window.display_handle().unwrap().as_raw(),
                window.window_handle().unwrap().as_raw(),
                None,
            )
            .unwrap()
        };

        Arc::new(PresentSurface {
            surface,
            surface_khr,
        })
    }

    pub fn new(
        device_manager: Arc<DeviceManager>,
        surface: Arc<PresentSurface>,
        frames_in_flight: u32,
        msaa_samples: u8,
    ) -> CrystalResult<Arc<Self>> {
        let swapchain = Swapchain::new(device_manager.clone(), surface.clone())?;

        Ok(Arc::new(Presentation {
            device_manager,
            surface,
            swapchain,

            frames_in_flight,
            msaa_samples,

            current_frame: Mutex::new(0),
            image_index: Mutex::new(0),
        }))
    }
}
