use std::{
    collections::VecDeque,
    iter::zip,
    sync::{Arc, Mutex, RwLock},
};

use ash::vk;

use crate::{
    Texture,
    debug::log,
    errors::{GraphicsError, GraphicsResult},
    traits,
    vulkan::{VulkanTexture, commands::CommandEntry},
};

use super::{
    depth::{DepthResources, find_depth_format},
    devices::DeviceManager,
    images::Image,
    sync::GpuSync,
};

pub struct VulkanRenderTarget {
    device_manager: Arc<DeviceManager>,
    pub(crate) command_entry: Arc<CommandEntry>,
    pub extent: RwLock<vk::Extent2D>,
    pub render_pass: vk::RenderPass,

    pub framebuffers: Vec<RwLock<vk::Framebuffer>>,
    pub color_images: RwLock<Vec<Arc<Image>>>,
    depth_resources: RwLock<Vec<DepthResources>>,

    pub msaa_samples: vk::SampleCountFlags,
    image_format: vk::Format,

    pub sync: Arc<Mutex<GpuSync>>,
    pub children: Mutex<VecDeque<Arc<VulkanRenderTarget>>>,
}

impl Drop for VulkanRenderTarget {
    fn drop(&mut self) {
        unsafe {
            self.destroy_framebuffers();

            self.device_manager
                .device
                .destroy_render_pass(self.render_pass, None);
        }
    }
}

impl traits::RenderTarget for VulkanRenderTarget {
    fn as_vulkan(self: Arc<Self>) -> Option<Arc<super::VulkanRenderTarget>> {
        Some(self)
    }

    fn create_render_target(
        &self,
        extent: [u32; 2],
        anisotropy_texels: f32,
        msaa_samples: u8,
    ) -> GraphicsResult<(Arc<dyn traits::RenderTarget>, Arc<dyn Texture>)> {
        let texture = VulkanTexture::new(self.device_manager.clone(), extent, anisotropy_texels)?;

        let extent = vk::Extent2D {
            width: extent[0],
            height: extent[1],
        };

        let render_target = VulkanRenderTarget::new(
            self.device_manager.clone(),
            texture.image.format,
            extent,
            vec![texture.image.image_view],
            msaa_samples,
            false,
        )?;

        self.children
            .lock()
            .unwrap()
            .push_back(render_target.clone());

        Ok((render_target, texture))
    }
}

impl VulkanRenderTarget {
    pub(crate) fn extent(&self) -> vk::Extent2D {
        *self.extent.read().unwrap()
    }

    fn destroy_framebuffers(&self) {
        unsafe {
            self.framebuffers.iter().for_each(|framebuffer| {
                self.device_manager
                    .device
                    .destroy_framebuffer(*framebuffer.read().unwrap(), None)
            });
        }
    }

    pub fn update_resources(
        &self,
        extent: vk::Extent2D,
        images: Vec<vk::ImageView>,
    ) -> GraphicsResult<()> {
        *self.extent.write().unwrap() = extent;
        self.destroy_framebuffers();
        let (framebuffers, depth_resources, color_images) = Self::create_resources(
            self.device_manager.clone(),
            self.image_format,
            extent,
            images,
            self.msaa_samples,
            self.render_pass,
        )?;

        zip(&self.framebuffers, framebuffers).for_each(|(framebuffer_left, framebuffer_right)| {
            *framebuffer_left.write().unwrap() = framebuffer_right
        });
        *self.depth_resources.write().unwrap() = depth_resources;
        *self.color_images.write().unwrap() = color_images;

        Ok(())
    }

    pub(crate) fn create_resources(
        device_manager: Arc<DeviceManager>,
        image_format: vk::Format,
        extent: vk::Extent2D,
        images: Vec<vk::ImageView>,
        samples: vk::SampleCountFlags,
        render_pass: vk::RenderPass,
    ) -> GraphicsResult<(Vec<vk::Framebuffer>, Vec<DepthResources>, Vec<Arc<Image>>)> {
        let mut framebuffers = vec![];
        let mut depth_resources = vec![];
        let mut color_images = vec![];

        for image_view in images {
            let depth_resource =
                DepthResources::new(device_manager.clone(), extent.width, extent.height, samples)?;

            let color_image = Image::new(
                device_manager.clone(),
                extent.width,
                extent.height,
                samples,
                image_format,
                vk::ImageTiling::OPTIMAL,
                vk::ImageAspectFlags::COLOR,
                vk::ImageUsageFlags::TRANSIENT_ATTACHMENT | vk::ImageUsageFlags::COLOR_ATTACHMENT,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
                false,
                1.,
            )?;

            let attachments = if samples != vk::SampleCountFlags::TYPE_1 {
                vec![
                    color_image.image_view,
                    depth_resource.image.image_view,
                    image_view,
                ]
            } else {
                vec![image_view, depth_resource.image.image_view]
            };

            depth_resources.push(depth_resource);

            let create_info = vk::FramebufferCreateInfo::default()
                .render_pass(render_pass)
                .attachments(&attachments)
                .width(extent.width)
                .height(extent.height)
                .layers(1);

            let framebuffer =
                match unsafe { device_manager.device.create_framebuffer(&create_info, None) } {
                    Ok(framebuffer) => framebuffer,
                    Err(e) => {
                        log!("cannot create framebuffer: {}", e);
                        return Err(GraphicsError::NotSupportedPresent);
                    }
                };
            framebuffers.push(framebuffer);
            color_images.push(color_image);
        }

        Ok((framebuffers, depth_resources, color_images))
    }

    pub(crate) fn new(
        device_manager: Arc<DeviceManager>,
        image_format: vk::Format,
        extent: vk::Extent2D,
        images: Vec<vk::ImageView>,
        msaa_samples: u8,
        present: bool,
    ) -> GraphicsResult<Arc<Self>> {
        let command_entry = if let Some(queue) = device_manager
            .queues
            .iter()
            .find(|queue| queue.flags.intersects(vk::QueueFlags::GRAPHICS))
        {
            CommandEntry::new(device_manager.clone(), queue.clone(), images.len() as u32)?
        } else {
            panic!("fatal: no graphics queue family!");
        };

        let counts = device_manager
            .device_properties
            .limits
            .framebuffer_color_sample_counts
            & device_manager
                .device_properties
                .limits
                .framebuffer_depth_sample_counts;

        let samples = match msaa_samples {
            2 => vk::SampleCountFlags::TYPE_2,
            4 => vk::SampleCountFlags::TYPE_4,
            8 => vk::SampleCountFlags::TYPE_8,
            16 => vk::SampleCountFlags::TYPE_16,
            32 => vk::SampleCountFlags::TYPE_32,
            64 => vk::SampleCountFlags::TYPE_64,
            _ => vk::SampleCountFlags::TYPE_1,
        };

        if counts & samples != samples {
            panic!(
                "fatal: device is not supported for sample count: {}",
                msaa_samples
            );
        };

        let initial_layout = vk::ImageLayout::UNDEFINED;
        let final_layout = if present {
            vk::ImageLayout::PRESENT_SRC_KHR
        } else {
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        };

        let color_attachment = vk::AttachmentDescription::default()
            .format(image_format)
            .samples(samples)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(initial_layout)
            .final_layout(if samples != vk::SampleCountFlags::TYPE_1 {
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
            } else {
                final_layout
            });

        let depth_attachment = vk::AttachmentDescription::default()
            .format(find_depth_format(
                device_manager.clone(),
                vk::ImageTiling::OPTIMAL,
                vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT,
            ))
            .samples(samples)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(initial_layout)
            .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        let color_attachment_resolve = vk::AttachmentDescription::default()
            .format(image_format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::DONT_CARE)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(initial_layout)
            .final_layout(final_layout);

        let mut attachments = vec![color_attachment, depth_attachment];

        if samples != vk::SampleCountFlags::TYPE_1 {
            attachments.push(color_attachment_resolve);
        }

        let color_attachment_reference = vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        let depth_attachment_reference = vk::AttachmentReference::default()
            .attachment(1)
            .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        let color_attachments = &[color_attachment_reference];

        let color_attachment_resolve_reference = vk::AttachmentReference::default()
            .attachment(2)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        let mut resolve_attachments = vec![];

        if samples != vk::SampleCountFlags::TYPE_1 {
            resolve_attachments.push(color_attachment_resolve_reference)
        }

        let mut subpass = vk::SubpassDescription::default()
            .color_attachments(color_attachments)
            .depth_stencil_attachment(&depth_attachment_reference)
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS);

        if samples != vk::SampleCountFlags::TYPE_1 {
            subpass = subpass.resolve_attachments(&resolve_attachments)
        }

        let subpasses = &[subpass];

        let dependency = vk::SubpassDependency::default()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::TOP_OF_PIPE)
            .dst_stage_mask(
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            )
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            );

        let end_dependency = vk::SubpassDependency::default()
            .src_subpass(0)
            .dst_subpass(vk::SUBPASS_EXTERNAL)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_stage_mask(vk::PipelineStageFlags::BOTTOM_OF_PIPE)
            .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
            .dst_access_mask(vk::AccessFlags::empty());

        let dependencies = &[dependency, end_dependency];

        let render_pass_create_info = vk::RenderPassCreateInfo::default()
            .attachments(&attachments)
            .subpasses(subpasses)
            .dependencies(dependencies);

        let render_pass = match unsafe {
            device_manager
                .device
                .create_render_pass(&render_pass_create_info, None)
        } {
            Ok(render_pass) => render_pass,
            Err(e) => {
                log!("failed to crate render pass: {}", e);
                return Err(GraphicsError::NotSupportedPresent);
            }
        };

        let sync = GpuSync::new(device_manager.clone(), images.len() as u32)?;

        let (framebuffers, depth_resources, color_images) = Self::create_resources(
            device_manager.clone(),
            image_format,
            extent,
            images,
            samples,
            render_pass,
        )?;

        Ok(Arc::new(Self {
            device_manager,
            command_entry,
            extent: RwLock::new(extent),
            render_pass,
            framebuffers: framebuffers
                .iter()
                .map(|framebuffer| RwLock::new(*framebuffer))
                .collect(),

            depth_resources: RwLock::new(depth_resources),
            color_images: RwLock::new(color_images),

            msaa_samples: samples,
            image_format,

            sync,
            children: Mutex::new(VecDeque::with_capacity(16)),
        }))
    }
}
