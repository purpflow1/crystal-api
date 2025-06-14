use std::{cell::Ref, ffi::CString, iter::zip, sync::Arc};

use ash::vk;

use crate::{
    ShaderStage,
    debug::log,
    errors::{CrystalError, CrystalResult},
    mesh::{Attribute, VertexTexture},
    shader::Shader,
    traits,
};

use super::{
    depth::{DepthResources, find_depth_format},
    devices::DeviceManager,
    images::Image,
};

pub struct VulkanRenderTarget {
    device_manager: Arc<DeviceManager>,
    pub extent: vk::Extent2D,
    pub render_pass: vk::RenderPass,

    pub framebuffers: Vec<vk::Framebuffer>,
    depth_resources: Vec<DepthResources>,

    pub msaa_samples: vk::SampleCountFlags,
}

impl traits::Pipeline for vk::Pipeline {
    fn as_vulkan_mut(&mut self) -> Option<&mut ash::vk::Pipeline> {
        Some(self)
    }

    fn as_vulkan_ref(&self) -> Option<&ash::vk::Pipeline> {
        Some(self)
    }
}

impl traits::RenderTarget for VulkanRenderTarget {
    fn update_size(&mut self, width: u32, height: u32) -> CrystalResult<()> {
        self.extent = vk::Extent2D { width, height };
        Ok(())
    }

    fn create_graphics_pipeline(
        &self,
        layout: Ref<dyn traits::Layout>,
        shaders: &[Shader],
        attributes: &[Attribute],
    ) -> CrystalResult<Arc<dyn traits::Pipeline>> {
        if shaders.is_empty() {
            log!("no shaders specified");
            return Err(CrystalError::ShaderError);
        }

        let mut shader_modules = Vec::new();

        let en = CString::new("main").unwrap();
        let entry_point_name = en.as_c_str();

        for shader in shaders {
            let shader_stage_flag = match shader.stage {
                ShaderStage::Vertex => vk::ShaderStageFlags::VERTEX,
                ShaderStage::Fragment => vk::ShaderStageFlags::FRAGMENT,
                ShaderStage::Geometry => vk::ShaderStageFlags::GEOMETRY,
            };

            let shader_module_create_info =
                vk::ShaderModuleCreateInfo::default().code(shader.code.as_words());

            let module = match unsafe {
                self.device_manager
                    .device
                    .create_shader_module(&shader_module_create_info, None)
            } {
                Ok(module) => module,
                Err(e) => {
                    log!("cannot create shader module: {}", e);
                    return Err(CrystalError::ShaderError);
                }
            };

            let shader_stage_create_info = vk::PipelineShaderStageCreateInfo::default()
                .stage(shader_stage_flag)
                .module(module)
                .name(entry_point_name);

            shader_modules.push(shader_stage_create_info);
        }

        let binding_descriptions = &[vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(size_of::<VertexTexture>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)];

        let mut attribute_descriptions = vec![];

        for (location, attribute) in zip(0..attributes.len() as u32, attributes) {
            let attribute_description = vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(location)
                .format(match attribute.size {
                    4 => vk::Format::R32_SFLOAT,
                    8 => vk::Format::R32G32_SFLOAT,
                    12 => vk::Format::R32G32B32_SFLOAT,
                    16 => vk::Format::R32G32B32A32_SFLOAT,
                    _ => vk::Format::R32G32B32_SFLOAT,
                })
                .offset(attribute.offset as u32);
            attribute_descriptions.push(attribute_description);
        }

        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(binding_descriptions)
            .vertex_attribute_descriptions(&attribute_descriptions);

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        let viewport = vk::Viewport::default()
            .x(0.)
            .y(0.)
            .width(self.extent.width as f32)
            .height(self.extent.height as f32)
            .min_depth(0.)
            .max_depth(1.);

        let scissor = vk::Rect2D::default().extent(self.extent);

        let dynamic_states = &[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];

        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(dynamic_states);

        let viewports = &[viewport];
        let scissors = &[scissor];

        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewports(viewports)
            .scissors(scissors);

        let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.)
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .depth_bias_enable(false);

        let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
            .sample_shading_enable(false)
            .rasterization_samples(self.msaa_samples);

        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
            .alpha_blend_op(vk::BlendOp::ADD);

        let attachments = &[color_blend_attachment];

        let color_blending = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .attachments(attachments);

        let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS)
            .depth_bounds_test_enable(false);

        let layout = match layout.as_vulkan_ref() {
            Some(layout) => layout,
            None => panic!("fatal: wrong layout type, expected vulkan"),
        };

        let pipeline_create_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&shader_modules)
            .vertex_input_state(&vertex_input_info)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisampling)
            .color_blend_state(&color_blending)
            .dynamic_state(&dynamic_state)
            .depth_stencil_state(&depth_stencil_state)
            .layout(layout.pipeline_layout)
            .render_pass(self.render_pass);

        match unsafe {
            self.device_manager.device.create_graphics_pipelines(
                vk::PipelineCache::null(),
                &[pipeline_create_info],
                None,
            )
        } {
            Ok(pipeline) => Ok(Arc::new(pipeline[0])),
            Err(es) => {
                log!("cannot create graphics pipeline: {}", es.1);
                Err(CrystalError::CannotCreateRenderPass)
            }
        }
    }
}

impl VulkanRenderTarget {
    pub(crate) fn new(
        instance: &ash::Instance,
        device_manager: Arc<DeviceManager>,
        image_format: vk::Format,
        extent: vk::Extent2D,
        image_views: &[vk::ImageView],
        msaa_samples: u8,
    ) -> CrystalResult<Self> {
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

        let color_attachment = vk::AttachmentDescription::default()
            .format(image_format)
            .samples(samples)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(if samples != vk::SampleCountFlags::TYPE_1 {
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
            } else {
                vk::ImageLayout::PRESENT_SRC_KHR
            });

        let depth_attachment = vk::AttachmentDescription::default()
            .format(find_depth_format(
                instance,
                device_manager.clone(),
                vk::ImageTiling::OPTIMAL,
                vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT,
            ))
            .samples(samples)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        let color_attachment_resolve = vk::AttachmentDescription::default()
            .format(image_format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::DONT_CARE)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(if samples != vk::SampleCountFlags::TYPE_1 {
                vk::ImageLayout::PRESENT_SRC_KHR
            } else {
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
            });

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
            .src_stage_mask(
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            )
            .src_access_mask(vk::AccessFlags::empty())
            .dst_stage_mask(
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            )
            .dst_access_mask(
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            );

        let dependencies = &[dependency];

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
                return Err(CrystalError::CannotCreateRenderPass);
            }
        };

        let mut framebuffers = vec![];
        let mut depth_resources = vec![];

        for image_view in image_views {
            let depth_resource = DepthResources::new(
                device_manager.clone(),
                instance,
                extent.width,
                extent.height,
                samples,
            )?;

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
                    *image_view,
                ]
            } else {
                vec![*image_view, depth_resource.image.image_view]
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
                        return Err(CrystalError::CannotCreateFramebuffer);
                    }
                };
            framebuffers.push(framebuffer);
        }

        Ok(Self {
            device_manager,
            extent,
            render_pass,
            framebuffers,

            depth_resources,

            msaa_samples: samples,
        })
    }
}
