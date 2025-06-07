use std::{ffi::CStr, sync::Arc};

use ash::{Instance, vk};

use crate::{
    debug::log,
    errors::{CrystalError, CrystalResult},
};

use super::presentation::Presentation;

pub(crate) struct DeviceManager {
    pub device: Arc<ash::Device>,
    pub physical_device: vk::PhysicalDevice,
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub device_properties: vk::PhysicalDeviceProperties,
    pub queue_families_indices: QueueFamiliesIndices,
}

impl DeviceManager {
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
}

#[derive(Clone, Copy, Default)]
pub struct QueueFamiliesIndices {
    pub graphics_index: Option<u32>,
    pub present_index: Option<u32>,
    pub compute_index: Option<u32>,
    pub transfer_index: Option<u32>,
}

impl QueueFamiliesIndices {
    pub fn get_unique_queue_families(&self) -> Vec<u32> {
        let mut unique_queue_families = vec![];

        match self.graphics_index {
            Some(ind) => unique_queue_families.push(ind),
            None => (),
        }
        match self.present_index {
            Some(ind) => {
                if unique_queue_families.iter().find(|&&x| x == ind).is_none() {
                    unique_queue_families.push(ind)
                }
            }
            None => (),
        }
        match self.compute_index {
            Some(ind) => {
                if unique_queue_families.iter().find(|&&x| x == ind).is_none() {
                    unique_queue_families.push(ind)
                }
            }
            None => (),
        }
        match self.transfer_index {
            Some(ind) => {
                if unique_queue_families.iter().find(|&&x| x == ind).is_none() {
                    unique_queue_families.push(ind)
                }
            }
            None => (),
        }
        unique_queue_families
    }
}

pub fn pick_physical_device(
    instance: &Instance,
    surface: Option<&Presentation>,
    extensions: &[*const i8],
) -> CrystalResult<(vk::PhysicalDevice, QueueFamiliesIndices)> {
    let devices = match unsafe { instance.enumerate_physical_devices() } {
        Ok(devices) => devices,
        Err(e) => {
            log!("cannot enumerate physical devices: {}", e);
            return Err(CrystalError::CannotPickPhysicalDevice);
        }
    };

    let mut picked_device = None;
    let mut picked_device_type = None;
    let mut device_name = "";

    'devloop: for device in devices {
        let props = unsafe { instance.get_physical_device_properties(device) };

        let extension_props =
            match unsafe { instance.enumerate_device_extension_properties(device) } {
                Ok(props) => props,
                Err(e) => {
                    log!("cannot enumerate device extension properties: {}", e);
                    continue;
                }
            };

        for &req_ext in extensions {
            let req_ext = unsafe { CStr::from_ptr(req_ext) }.to_str().unwrap();
            if extension_props
                .iter()
                .find(|ext| ext.extension_name_as_c_str().unwrap().to_str().unwrap() == req_ext)
                .is_none()
            {
                continue 'devloop;
            }
        }

        match props.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => {
                picked_device = Some(device);
                device_name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) }
                    .to_str()
                    .unwrap();

                break;
            }
            vk::PhysicalDeviceType::INTEGRATED_GPU => {
                picked_device_type = Some(props.device_type);
                picked_device = Some(device);
                device_name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) }
                    .to_str()
                    .unwrap();
            }
            _ => match picked_device_type {
                None => picked_device = Some(device),
                Some(_) => (),
            },
        }
    }

    let device = match picked_device {
        Some(device) => {
            log!("picked device: {}", device_name);
            device
        }
        None => {
            log!("no suitable devices found");
            return Err(CrystalError::CannotPickPhysicalDevice);
        }
    };

    let queue_families_indices = find_queue_families(instance, surface, device);

    Ok((device, queue_families_indices))
}

fn find_queue_families(
    instance: &Instance,
    surface: Option<&Presentation>,
    device: vk::PhysicalDevice,
) -> QueueFamiliesIndices {
    let mut queue_families = QueueFamiliesIndices::default();

    let props = unsafe { instance.get_physical_device_queue_family_properties(device) };
    for (index, family) in props.iter().filter(|f| f.queue_count > 0).enumerate() {
        let index = index as u32;

        if family.queue_flags.contains(vk::QueueFlags::GRAPHICS)
            && queue_families.graphics_index.is_none()
        {
            queue_families.graphics_index = Some(index);
        }

        if family.queue_flags.contains(vk::QueueFlags::COMPUTE)
            && queue_families.compute_index.is_none()
        {
            queue_families.compute_index = Some(index);
        }

        if family.queue_flags.contains(vk::QueueFlags::TRANSFER)
            && queue_families.transfer_index.is_none()
        {
            queue_families.transfer_index = Some(index);
        }

        let present_support = unsafe {
            match surface {
                Some(surface) => surface
                    .surface
                    .get_physical_device_surface_support(device, index, surface.surface_khr)
                    .unwrap(),
                None => false,
            }
        };

        if present_support && queue_families.present_index.is_none() {
            queue_families.present_index = Some(index);
        }
    }

    queue_families
}

pub fn create_logical_device(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    queue_families_indices: &QueueFamiliesIndices,
    extensions: &[*const i8],
    features: vk::PhysicalDeviceFeatures,
) -> CrystalResult<ash::Device> {
    let mut queues_create_infos = vec![];
    let unique_queue_families = queue_families_indices.get_unique_queue_families();

    for idx in unique_queue_families {
        queues_create_infos.push(
            vk::DeviceQueueCreateInfo::default()
                .queue_family_index(idx)
                .queue_priorities(&[1.]),
        )
    }

    let device_create_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queues_create_infos)
        .enabled_features(&features)
        .enabled_extension_names(extensions);

    match unsafe { instance.create_device(physical_device, &device_create_info, None) } {
        Err(e) => {
            log!("cannot create logical device: {}", e);
            Err(CrystalError::CannotInitDevice)
        }
        Ok(device) => Ok(device),
    }
}
