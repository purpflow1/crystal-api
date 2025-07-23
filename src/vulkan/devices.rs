use std::{
    ffi::CStr,
    sync::{Arc, Mutex, MutexGuard},
};

use ash::{Instance, vk};

use crate::{
    debug::log,
    errors::{GraphicsError, GraphicsResult},
    vulkan::presentation::PresentSurface,
};

#[derive(Clone, Default)]
pub struct PhysicalDeviceExtensions {
    pub swapchain_compression: bool,
    pub present_support: bool,
}

pub struct Queue {
    pub device: Arc<ash::Device>,
    pub flags: vk::QueueFlags,
    pub present_support: bool,
    pub family_index: u32,
    handle: Mutex<vk::Queue>,
}

impl std::fmt::Debug for Queue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "Queue:\n\
            family  = {}\n\
            flags   = {:?}\n\
            present = {}",
            self.family_index, self.flags, self.present_support
        ))
    }
}

impl Queue {
    fn new(
        device: Arc<ash::Device>,
        queue_families: &[(vk::QueueFlags, QueueFamilyInfo)],
    ) -> Vec<Arc<Queue>> {
        queue_families
            .iter()
            .map(|(flags, info)| {
                (0..info.queue_count)
                    .map(|idx| {
                        Arc::new(Queue {
                            device: device.clone(),
                            flags: *flags,
                            handle: Mutex::new(unsafe {
                                device.get_device_queue(info.family, idx)
                            }),
                            present_support: info.present_support,
                            family_index: info.family,
                        })
                    })
                    .collect::<Vec<Arc<Queue>>>()
            })
            .collect::<Vec<Vec<Arc<Queue>>>>()
            .concat()
    }

    pub fn wait_idle(&self) -> GraphicsResult<()> {
        match unsafe { self.device.queue_wait_idle(*self.handle.lock().unwrap()) } {
            Ok(()) => Ok(()),
            Err(e) => {
                log!("queue wait idle error: {:?}", e);
                Err(GraphicsError::SyncError)
            }
        }
    }

    pub fn submit(&self, submits: &[vk::SubmitInfo<'_>], fence: vk::Fence) -> GraphicsResult<()> {
        let lock = self.handle.lock().unwrap();

        if let Err(e) = unsafe { self.device.queue_submit(*lock, submits, fence) } {
            log!("queue submit error: {:?}", e);
            return Err(GraphicsError::SyncError);
        }

        Ok(())
    }

    pub fn submit_still_lock(
        &self,
        submits: &[vk::SubmitInfo<'_>],
        fence: vk::Fence,
    ) -> GraphicsResult<MutexGuard<vk::Queue>> {
        let lock = self.handle.lock().unwrap();

        if let Err(e) = unsafe { self.device.queue_submit(*lock, submits, fence) } {
            log!("queue submit error: {:?}", e);
            return Err(GraphicsError::SyncError);
        }

        Ok(lock)
    }
}

pub struct DeviceManager {
    pub entry: Arc<ash::Entry>,
    pub instance: Arc<ash::Instance>,
    pub device: Arc<ash::Device>,
    pub physical_device: vk::PhysicalDevice,
    pub device_name: String,
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub device_properties: vk::PhysicalDeviceProperties,
    pub queues: Vec<Arc<Queue>>,
    pub supported_extensions: Vec<String>,
}

impl Drop for DeviceManager {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

impl DeviceManager {
    pub fn wait_idle(&self) -> GraphicsResult<()> {
        let locks: Vec<MutexGuard<vk::Queue>> = self
            .queues
            .iter()
            .map(|queue| queue.handle.lock().unwrap())
            .collect();

        if let Err(e) = unsafe { self.device.device_wait_idle() } {
            log!("cannot device wait idle: {:?}", e);
            return Err(GraphicsError::SyncError);
        }

        drop(locks);

        Ok(())
    }

    pub fn find_memory_type_index(
        &self,
        flags: vk::MemoryPropertyFlags,
        type_filter: u32,
    ) -> GraphicsResult<u32> {
        for i in 0..self.memory_properties.memory_type_count {
            if (type_filter & (1 << i)) != 0
                && (self.memory_properties.memory_types[i as usize].property_flags & flags) == flags
            {
                return Ok(i);
            }
        }

        log!("cannot find suitable memory type");
        Err(GraphicsError::MemoryError)
    }

    pub fn new(
        entry: Arc<ash::Entry>,
        instance: Arc<Instance>,
        surface: Option<Arc<PresentSurface>>,
    ) -> GraphicsResult<Arc<Self>> {
        let (physical_device, device_name) =
            pick_physical_device(instance.clone(), surface.is_none())?;

        let supported_extensions = query_extensions_support(instance.clone(), physical_device)?;

        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };

        let device_properties = unsafe { instance.get_physical_device_properties(physical_device) };

        let queue_families = find_queue_families(instance.clone(), surface, physical_device);

        if queue_families.len() == 0 {
            return Err(GraphicsError::NotSupportedDevice);
        }

        let (logical_device, queues) = create_logical_device(
            instance.clone(),
            physical_device,
            &queue_families,
            &supported_extensions,
            vk::PhysicalDeviceFeatures::default().sampler_anisotropy(true),
        )?;

        Ok(Arc::new(Self {
            entry,
            instance,
            device: logical_device.clone(),
            physical_device,
            device_name,
            memory_properties,
            device_properties,
            queues,
            supported_extensions: supported_extensions
                .iter()
                .map(|ext| ext.to_str().unwrap().to_string())
                .collect(),
        }))
    }
}

struct QueueFamilyInfo {
    family: u32,
    queue_count: u32,
    present_support: bool,
}

impl std::fmt::Debug for QueueFamilyInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "QueueFamilyInfo:\n\
            family      = {}\n\
            queue_count = {}\n\
            present     = {}",
            self.family, self.queue_count, self.present_support
        ))
    }
}

fn query_extensions_support<'a>(
    instance: Arc<Instance>,
    device: vk::PhysicalDevice,
) -> GraphicsResult<Vec<&'a CStr>> {
    let mut supported_extensions = Vec::with_capacity(64);

    let extensions = [
        vk::EXT_IMAGE_COMPRESSION_CONTROL_NAME,
        vk::EXT_IMAGE_COMPRESSION_CONTROL_SWAPCHAIN_NAME,
        vk::KHR_SWAPCHAIN_NAME,
    ];

    let extension_props = match unsafe { instance.enumerate_device_extension_properties(device) } {
        Ok(props) => props,
        Err(e) => {
            log!("cannot enumerate device extension properties: {}", e);
            return Err(GraphicsError::ConnotInitLibrary);
        }
    };

    let device_supported_extensions: Vec<&CStr> = extension_props
        .iter()
        .map(|ext| ext.extension_name_as_c_str().unwrap())
        .collect();

    for req_ext in &extensions {
        if device_supported_extensions
            .iter()
            .find(|&ext| *ext == *req_ext)
            .is_some()
        {
            supported_extensions.push(*req_ext)
        }
    }

    Ok(supported_extensions)
}

fn pick_physical_device<'a>(
    instance: Arc<Instance>,
    get_first: bool,
) -> GraphicsResult<(vk::PhysicalDevice, String)> {
    let devices = match unsafe { instance.enumerate_physical_devices() } {
        Ok(devices) => devices,
        Err(e) => {
            log!("cannot enumerate physical devices: {}", e);
            return Err(GraphicsError::NotSupportedDevice);
        }
    };

    if devices.is_empty() {
        log!("No devices found!");
        return Err(GraphicsError::ConnotInitLibrary);
    }

    if get_first {
        let device = devices[0];

        let props = unsafe { instance.get_physical_device_properties(device) };
        let device_name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) }
            .to_str()
            .unwrap();

        return Ok((device, device_name.to_string()));
    }

    let mut found = Err(GraphicsError::NotSupportedDevice);

    for device in devices {
        let props = unsafe { instance.get_physical_device_properties(device) };

        let device_name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) }
            .to_str()
            .unwrap()
            .to_string();

        match props.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => return Ok((device, device_name)),
            vk::PhysicalDeviceType::INTEGRATED_GPU => found = Ok((device, device_name)),
            _ => {
                if found.is_err() {
                    found = Ok((device, device_name))
                }
            }
        }
    }

    return found;
}

fn find_queue_families(
    instance: Arc<Instance>,
    surface: Option<Arc<PresentSurface>>,
    device: vk::PhysicalDevice,
) -> Vec<(vk::QueueFlags, QueueFamilyInfo)> {
    let mut queue_families = Vec::<(vk::QueueFlags, QueueFamilyInfo)>::new();

    let props = unsafe { instance.get_physical_device_queue_family_properties(device) };
    for (index, family) in props.iter().filter(|f| f.queue_count > 0).enumerate() {
        let index = index as u32;

        let present_support = unsafe {
            match surface.clone() {
                Some(surface) => surface
                    .surface
                    .get_physical_device_surface_support(device, index, surface.surface_khr)
                    .unwrap(),
                None => false,
            }
        };

        let mut to_push = false;

        queue_families.iter_mut().for_each(|(flags, info)| {
            if (*info).family == index {
                *flags |= family.queue_flags;
                info.present_support = present_support
            } else if !flags.intersects(family.queue_flags) {
                to_push = true;
            }
        });

        if to_push || queue_families.is_empty() {
            queue_families.push((
                family.queue_flags,
                QueueFamilyInfo {
                    family: index,
                    queue_count: family.queue_count,
                    present_support,
                },
            ));
        }
    }

    queue_families
}

fn create_logical_device(
    instance: Arc<Instance>,
    physical_device: vk::PhysicalDevice,
    queue_families: &[(vk::QueueFlags, QueueFamilyInfo)],
    extensions: &[&CStr],
    features: vk::PhysicalDeviceFeatures,
) -> GraphicsResult<(Arc<ash::Device>, Vec<Arc<Queue>>)> {
    let mut queues_create_infos = vec![];

    let priorities = vec![1.; 256];

    for (_, info) in queue_families {
        queues_create_infos.push(
            vk::DeviceQueueCreateInfo::default()
                .queue_family_index(info.family)
                .queue_priorities(&priorities[0..info.queue_count as usize]),
        )
    }

    let extension_names: Vec<_> = extensions.iter().map(|ext| ext.as_ptr()).collect();

    let device_create_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queues_create_infos)
        .enabled_features(&features)
        .enabled_extension_names(&extension_names);

    let device = match unsafe { instance.create_device(physical_device, &device_create_info, None) }
    {
        Err(e) => {
            log!("cannot create logical device: {}", e);
            return Err(GraphicsError::NotSupportedDevice);
        }
        Ok(device) => Arc::new(device),
    };

    let queues = Queue::new(device.clone(), queue_families);

    Ok((device, queues))
}
