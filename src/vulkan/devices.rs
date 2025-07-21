use std::{
    ffi::CStr,
    sync::{Arc, Mutex, MutexGuard},
};

use ash::{Instance, vk};

use crate::{
    debug::log,
    errors::{CrystalError, CrystalResult},
    vulkan::presentation::PresentSurface,
};

#[derive(Clone, Default)]
pub struct PhysicalDeviceExtensions {
    pub compression: bool,
    pub formats_4444: bool,
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

    pub fn wait_idle(&self) -> CrystalResult<()> {
        match unsafe { self.device.queue_wait_idle(*self.handle.lock().unwrap()) } {
            Ok(()) => Ok(()),
            Err(e) => {
                log!("queue wait idle error: {:?}", e);
                Err(CrystalError::SyncError)
            }
        }
    }

    pub fn submit(&self, submits: &[vk::SubmitInfo<'_>], fence: vk::Fence) -> CrystalResult<()> {
        let lock = self.handle.lock().unwrap();

        if let Err(e) = unsafe { self.device.queue_submit(*lock, submits, fence) } {
            log!("queue submit error: {:?}", e);
            return Err(CrystalError::SyncError);
        }

        Ok(())
    }

    pub fn submit_still_lock(
        &self,
        submits: &[vk::SubmitInfo<'_>],
        fence: vk::Fence,
    ) -> CrystalResult<MutexGuard<vk::Queue>> {
        let lock = self.handle.lock().unwrap();

        if let Err(e) = unsafe { self.device.queue_submit(*lock, submits, fence) } {
            log!("queue submit error: {:?}", e);
            return Err(CrystalError::SyncError);
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
    pub extensions: PhysicalDeviceExtensions,
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
    pub fn wait_idle(&self) -> CrystalResult<()> {
        let locks: Vec<MutexGuard<vk::Queue>> = self
            .queues
            .iter()
            .map(|queue| queue.handle.lock().unwrap())
            .collect();

        if let Err(e) = unsafe { self.device.device_wait_idle() } {
            log!("cannot device wait idle: {:?}", e);
            return Err(CrystalError::SyncError);
        }

        drop(locks);

        Ok(())
    }

    pub fn find_memory_type_index(
        &self,
        flags: vk::MemoryPropertyFlags,
        type_filter: u32,
    ) -> CrystalResult<u32> {
        for i in 0..self.memory_properties.memory_type_count {
            if (type_filter & (1 << i)) != 0
                && (self.memory_properties.memory_types[i as usize].property_flags & flags) == flags
            {
                return Ok(i);
            }
        }

        log!("cannot find suitable memory type");
        Err(CrystalError::MemoryError)
    }

    pub fn new(
        entry: Arc<ash::Entry>,
        instance: Arc<Instance>,
        surface: Option<Arc<PresentSurface>>,
        extensions: &[*const i8],
    ) -> CrystalResult<Arc<Self>> {
        let (physical_device, device_name, physical_device_extensions) =
            pick_physical_device(&instance, extensions)?;

        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };

        let device_properties = unsafe { instance.get_physical_device_properties(physical_device) };

        let queue_families = find_queue_families(instance.clone(), surface, physical_device);

        if queue_families.len() == 0 {
            return Err(CrystalError::GpuIsNotSupported);
        }

        let (logical_device, queues) = create_logical_device(
            instance.clone(),
            physical_device,
            &queue_families,
            &extensions,
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
            extensions: physical_device_extensions,
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

fn pick_physical_device<'a>(
    instance: &Instance,
    extensions: &[*const i8],
) -> CrystalResult<(vk::PhysicalDevice, String, PhysicalDeviceExtensions)> {
    let devices = match unsafe { instance.enumerate_physical_devices() } {
        Ok(devices) => devices,
        Err(e) => {
            log!("cannot enumerate physical devices: {}", e);
            return Err(CrystalError::Unsupported);
        }
    };

    let extensions: Vec<&str> = extensions
        .iter()
        .map(|ext| unsafe { CStr::from_ptr(*ext) }.to_str().unwrap())
        .collect();

    let mut picked_device = None;
    let mut picked_device_type = None;
    let mut device_name = "";
    let mut device_supported_extensions = vec![];
    #[allow(unused)]
    let mut extension_props = vec![];

    'devloop: for device in devices {
        let props = unsafe { instance.get_physical_device_properties(device) };
        device_name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) }
            .to_str()
            .unwrap();
        let (api_version_maj, api_version_min, api_version_pat) = (
            vk::api_version_major(props.api_version),
            vk::api_version_minor(props.api_version),
            vk::api_version_patch(props.api_version),
        );

        extension_props = match unsafe { instance.enumerate_device_extension_properties(device) } {
            Ok(props) => props,
            Err(e) => {
                log!("cannot enumerate device extension properties: {}", e);
                continue;
            }
        };

        device_supported_extensions = extension_props
            .iter()
            .map(|ext| ext.extension_name_as_c_str().unwrap().to_str().unwrap())
            .collect();

        for req_ext in &extensions {
            if device_supported_extensions
                .iter()
                .find(|&ext| *ext == *req_ext)
                .is_none()
            {
                let version_name = format!(
                    "{}.{}.{}",
                    api_version_maj, api_version_min, api_version_pat
                );
                log!(
                    "{} with Vulkan API version {} does not have support for {}",
                    device_name,
                    version_name,
                    req_ext
                );
                continue 'devloop;
            }
        }

        match props.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => {
                picked_device = Some(device);
                break;
            }
            vk::PhysicalDeviceType::INTEGRATED_GPU => {
                picked_device_type = Some(props.device_type);
                picked_device = Some(device);
            }
            _ => match picked_device_type {
                None => picked_device = Some(device),
                Some(_) => (),
            },
        }
    }

    let device = match picked_device {
        Some(device) => device,
        None => {
            return Err(CrystalError::Unsupported);
        }
    };

    let mut supported_extensions = PhysicalDeviceExtensions::default();

    {
        let compression_extension = vk::EXT_IMAGE_COMPRESSION_CONTROL_NAME.as_ptr();
        let ext_name = unsafe { CStr::from_ptr(compression_extension) }
            .to_str()
            .unwrap();
        supported_extensions.compression = device_supported_extensions
            .iter()
            .find(|&dev_ext| *dev_ext == ext_name)
            .is_some();
    }

    {
        let formats_extension = vk::EXT_4444_FORMATS_NAME.as_ptr();
        let ext_name = unsafe { CStr::from_ptr(formats_extension) }
            .to_str()
            .unwrap();
        supported_extensions.formats_4444 = device_supported_extensions
            .iter()
            .find(|&dev_ext| *dev_ext == ext_name)
            .is_some();
    }

    Ok((device, device_name.to_string(), supported_extensions))
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
    extensions: &[*const i8],
    features: vk::PhysicalDeviceFeatures,
) -> CrystalResult<(Arc<ash::Device>, Vec<Arc<Queue>>)> {
    let mut queues_create_infos = vec![];

    let priorities = vec![1.; 256];

    for (_, info) in queue_families {
        queues_create_infos.push(
            vk::DeviceQueueCreateInfo::default()
                .queue_family_index(info.family)
                .queue_priorities(&priorities[0..info.queue_count as usize]),
        )
    }

    let device_create_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queues_create_infos)
        .enabled_features(&features)
        .enabled_extension_names(extensions);

    let device = match unsafe { instance.create_device(physical_device, &device_create_info, None) }
    {
        Err(e) => {
            log!("cannot create logical device: {}", e);
            return Err(CrystalError::CannotInitDevice);
        }
        Ok(device) => Arc::new(device),
    };

    let queues = Queue::new(device.clone(), queue_families);

    Ok((device, queues))
}
