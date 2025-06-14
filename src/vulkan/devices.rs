use std::{collections::BTreeMap, iter::zip, sync::Arc};

use crate::{
    debug::log,
    errors::{CrystalError, CrystalResult},
    vulkan::presentation::Presentation,
};

pub fn pick_physical_device(
    instance: Arc<vulkano::instance::Instance>,
    extensions: &vulkano::device::DeviceExtensions,
) -> CrystalResult<Arc<vulkano::device::physical::PhysicalDevice>> {
    let devices = match instance.enumerate_physical_devices() {
        Ok(devices) => devices,
        Err(e) => {
            log!("cannot enumerate physical devices: {:?}", e);
            return Err(CrystalError::CannotPickPhysicalDevice);
        }
    };

    let mut picked_device = None;
    let mut picked_device_type = None;
    let mut device_name = String::new();

    for device in devices {
        let props = device.properties();
        let supported_extensions = device.supported_extensions();

        if extensions.intersection(supported_extensions) != *extensions {
            continue;
        }

        match props.device_type {
            vulkano::device::physical::PhysicalDeviceType::DiscreteGpu => {
                picked_device = Some(device.clone());
                device_name = props.device_name.clone();

                break;
            }
            vulkano::device::physical::PhysicalDeviceType::IntegratedGpu => {
                picked_device_type = Some(props.device_type);
                picked_device = Some(device.clone());
                device_name = props.device_name.clone();
            }
            _ => match picked_device_type {
                None => {
                    picked_device_type = Some(props.device_type);
                    picked_device = Some(device.clone());
                    device_name = props.device_name.clone()
                }
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

    Ok(device)
}

pub fn create_logical_device(
    physical_device: Arc<vulkano::device::physical::PhysicalDevice>,
    surface: Option<&Presentation>,
    extensions: &vulkano::device::DeviceExtensions,
    features: &vulkano::device::DeviceFeatures,
) -> CrystalResult<(
    Arc<vulkano::device::Device>,
    Vec<(Arc<vulkano::device::Queue>, vulkano::device::QueueFlags)>,
)> {
    let mut queue_create_infos = vec![];
    let mut unique_queue_families_indices: BTreeMap<u32, vulkano::device::QueueFlags> =
        BTreeMap::new();

    for (index, props) in physical_device.queue_family_properties().iter().enumerate() {
        let index = index as u32;

        if props.queue_flags.intersects(
            vulkano::device::QueueFlags::COMPUTE | vulkano::device::QueueFlags::TRANSFER,
        ) {
            unique_queue_families_indices.insert(index, props.queue_flags);
        } else if props
            .queue_flags
            .intersects(vulkano::device::QueueFlags::GRAPHICS)
        {
            match surface {
                Some(presentation) => {
                    if props
                        .queue_flags
                        .contains(vulkano::device::QueueFlags::GRAPHICS)
                    {
                        let surface_support = physical_device
                            .surface_support(index, &presentation.surface)
                            .expect("cannot query surface support");

                        if !surface_support {
                            log!("present is not supported by device");
                            return Err(CrystalError::CannotInitDevice);
                        }

                        unique_queue_families_indices.insert(index, props.queue_flags);
                    }
                }
                None => (),
            }
        }
    }

    for (queue_family_index, _) in &unique_queue_families_indices {
        queue_create_infos.push(vulkano::device::QueueCreateInfo {
            flags: vulkano::device::QueueCreateFlags::empty(),
            queue_family_index: *queue_family_index,
            queues: vec![1.],
            ..Default::default()
        })
    }

    let device_create_info = vulkano::device::DeviceCreateInfo {
        queue_create_infos: queue_create_infos,
        enabled_features: *features,
        enabled_extensions: *extensions,
        ..Default::default()
    };

    let (device, queues) = match vulkano::device::Device::new(physical_device, device_create_info) {
        Err(e) => {
            log!("cannot create logical device: {:?}", e);
            return Err(CrystalError::CannotInitDevice);
        }
        Ok(device) => device,
    };

    return Ok((
        device,
        zip(
            // TODO make more readable
            queues.collect::<Vec<Arc<vulkano::device::Queue>>>(),
            unique_queue_families_indices
                .into_iter()
                .map(|(_, flag)| flag)
                .collect::<Vec<vulkano::device::QueueFlags>>(),
        )
        .collect(),
    ));
}
