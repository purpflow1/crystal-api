use std::{cell::Ref, iter::zip, sync::Arc};

use foldhash::{HashSet, HashSetExt};
use smallvec::{SmallVec, smallvec};
use vulkano::{
    device::DeviceOwned,
    format::{Format, FormatFeatures},
    image::{
        ImageAspects, ImageLayout, ImageTiling, ImageUsage, SampleCount, SampleCounts,
        view::ImageView,
    },
    memory::allocator::StandardMemoryAllocator,
    pipeline::{
        DynamicState, GraphicsPipeline, PipelineShaderStageCreateInfo,
        graphics::{
            GraphicsPipelineCreateInfo,
            color_blend::{
                AttachmentBlend, ColorBlendAttachmentState, ColorBlendState, ColorComponents,
            },
            depth_stencil::{DepthState, DepthStencilState},
            input_assembly::{InputAssemblyState, PrimitiveTopology},
            multisample::MultisampleState,
            rasterization::{CullMode, FrontFace, PolygonMode, RasterizationState},
            subpass::PipelineSubpassType,
            vertex_input::{
                VertexInputAttributeDescription, VertexInputBindingDescription, VertexInputRate,
                VertexInputState,
            },
            viewport::{Scissor, Viewport, ViewportState},
        },
    },
    render_pass::{
        AttachmentDescription, AttachmentLoadOp, AttachmentReference, AttachmentStoreOp,
        Framebuffer, FramebufferCreateInfo, RenderPass, RenderPassCreateInfo, SubpassDependency,
        SubpassDescription,
    },
    shader::{ShaderModule, ShaderModuleCreateInfo, spirv},
    sync::{AccessFlags, PipelineStages},
};

use crate::{
    debug::log,
    errors::{CrystalError, CrystalResult},
    mesh::{Attribute, VertexTexture},
    shader::Shader,
    traits,
};

use super::{depth::find_depth_format, images::Image};

pub struct VulkanRenderTarget {
    device: Arc<vulkano::device::Device>,
    pub render_pass: Arc<RenderPass>,
    pub framebuffers: Vec<Arc<Framebuffer>>,
    pub frames_in_flight: usize,
    pub current_frame: usize,
    pub extent: [u32; 2],
    samples: SampleCount,
}

impl traits::Pipeline for GraphicsPipeline {
    fn as_vulkan(self: Arc<GraphicsPipeline>) -> Option<Arc<GraphicsPipeline>> {
        Some(self.clone())
    }
}

impl traits::RenderTarget for VulkanRenderTarget {
    fn as_vulkan_mut(&mut self) -> Option<&mut VulkanRenderTarget> {
        Some(self)
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

        let layout = match layout.as_vulkan_ref() {
            Some(layout) => layout,
            None => panic!("fatal: wrong layout type, expected vulkan"),
        };

        let mut shader_modules = SmallVec::new();

        for shader in shaders {
            let slice = shader.code.as_slice();
            let code = spirv::bytes_to_words(slice).unwrap().into_owned();
            let create_info = ShaderModuleCreateInfo::new(&code);

            let module = match unsafe { ShaderModule::new(self.device.clone(), create_info) } {
                Ok(module) => module,
                Err(e) => {
                    log!("cannot create shader module: {:?}", e);
                    return Err(CrystalError::ShaderError);
                }
            };

            let shader_stage_create_info = PipelineShaderStageCreateInfo::new(
                module
                    .entry_point("main")
                    .expect("fatal: no main entry point in SPIR-V shader"),
            );

            shader_modules.push(shader_stage_create_info);
        }

        let mut vertex_attributes: Vec<(u32, VertexInputAttributeDescription)> = vec![];

        for (location, attribute) in zip(0..attributes.len() as u32, attributes) {
            let attribute_description = VertexInputAttributeDescription {
                binding: location,
                format: match attribute.size {
                    4 => Format::R32_SFLOAT,
                    8 => Format::R32G32_SFLOAT,
                    12 => Format::R32G32B32_SFLOAT,
                    16 => Format::R32G32B32A32_SFLOAT,
                    _ => Format::R32G32B32_SFLOAT,
                },
                offset: attribute.offset as u32,
                ..Default::default()
            };

            vertex_attributes.push((location, attribute_description));
        }

        let binding = VertexInputBindingDescription {
            stride: size_of::<VertexTexture>() as u32,
            input_rate: VertexInputRate::Vertex,
            ..Default::default()
        };

        let vertex_input = VertexInputState::new()
            .binding(0, binding)
            .attributes(vertex_attributes);

        let input_assembly = InputAssemblyState {
            topology: PrimitiveTopology::TriangleList,
            primitive_restart_enable: false,
            ..Default::default()
        };

        let mut dynamic_states = HashSet::new();
        dynamic_states.insert(DynamicState::Viewport);
        dynamic_states.insert(DynamicState::Scissor);

        let viewport_state = ViewportState {
            viewports: smallvec![Viewport {
                extent: [self.extent[0] as f32, self.extent[1] as f32],
                ..Default::default()
            }],
            scissors: smallvec![Scissor {
                extent: self.extent,
                ..Default::default()
            }],
            ..Default::default()
        };

        let rasterizer = RasterizationState {
            depth_clamp_enable: false,
            rasterizer_discard_enable: false,
            depth_bias: None,
            polygon_mode: PolygonMode::Fill,
            line_width: 1.,
            cull_mode: CullMode::Back,
            front_face: FrontFace::CounterClockwise,
            ..Default::default()
        };

        let multisampling = MultisampleState {
            sample_shading: None,
            rasterization_samples: self.samples,
            ..Default::default()
        };

        let color_blend_attachment = ColorBlendAttachmentState {
            color_write_mask: ColorComponents::all(),
            blend: Some(AttachmentBlend::alpha()),
            ..Default::default()
        };

        let color_blending = ColorBlendState {
            logic_op: None,
            attachments: vec![color_blend_attachment],
            ..Default::default()
        };

        let depth_stencil_state = DepthStencilState {
            depth: Some(DepthState::simple()),
            ..Default::default()
        };

        let mut create_info = GraphicsPipelineCreateInfo::layout(layout.pipeline_layout.clone());
        create_info.stages = shader_modules;
        create_info.vertex_input_state = Some(vertex_input);
        create_info.input_assembly_state = Some(input_assembly);
        create_info.viewport_state = Some(viewport_state);
        create_info.rasterization_state = Some(rasterizer);
        create_info.multisample_state = Some(multisampling);
        create_info.color_blend_state = Some(color_blending);
        create_info.dynamic_state = dynamic_states;
        create_info.depth_stencil_state = Some(depth_stencil_state);
        create_info.subpass = Some(PipelineSubpassType::BeginRenderPass(
            self.render_pass.clone().first_subpass(),
        ));

        let pipeline = match GraphicsPipeline::new(self.device.clone(), None, create_info) {
            Ok(pipeline) => pipeline,
            Err(e) => {
                log!("cannot create graphics pipeline: {:?}", e);
                return Err(CrystalError::CannotCreateRenderPass);
            }
        };

        Ok(pipeline)
    }

    fn update_size(&mut self, extent: [u32; 2]) -> CrystalResult<()> {
        self.extent = extent;
        unimplemented!()
    }
}

impl VulkanRenderTarget {
    pub(crate) fn create_render_pass(
        device: Arc<vulkano::device::Device>,
        image_format: Format,
        msaa_samples: u8,
    ) -> CrystalResult<Arc<RenderPass>> {
        let counts = device
            .physical_device()
            .properties()
            .framebuffer_color_sample_counts
            & device
                .physical_device()
                .properties()
                .framebuffer_depth_sample_counts;

        let samples = match msaa_samples {
            2 => SampleCounts::SAMPLE_2,
            4 => SampleCounts::SAMPLE_4,
            8 => SampleCounts::SAMPLE_8,
            16 => SampleCounts::SAMPLE_16,
            32 => SampleCounts::SAMPLE_32,
            64 => SampleCounts::SAMPLE_64,
            _ => SampleCounts::SAMPLE_1,
        };

        if !counts.contains(samples) {
            panic!(
                "fatal: device is not supported for sample count: {}",
                msaa_samples
            );
        };

        let samples = match msaa_samples {
            2 => SampleCount::Sample2,
            4 => SampleCount::Sample4,
            8 => SampleCount::Sample8,
            16 => SampleCount::Sample16,
            32 => SampleCount::Sample32,
            64 => SampleCount::Sample64,
            _ => SampleCount::Sample1,
        };

        let color_attachment = AttachmentDescription {
            format: image_format,
            samples: samples,
            load_op: AttachmentLoadOp::Clear,
            store_op: AttachmentStoreOp::Store,
            stencil_load_op: Some(AttachmentLoadOp::DontCare),
            stencil_store_op: Some(AttachmentStoreOp::DontCare),
            initial_layout: ImageLayout::Undefined,
            final_layout: if samples != SampleCount::Sample1 {
                ImageLayout::ColorAttachmentOptimal
            } else {
                ImageLayout::PresentSrc
            },
            ..Default::default()
        };

        let depth_attachment = AttachmentDescription {
            format: find_depth_format(
                device.clone(),
                ImageTiling::Optimal,
                FormatFeatures::DEPTH_STENCIL_ATTACHMENT,
            ),
            samples: samples,
            load_op: AttachmentLoadOp::Clear,
            store_op: AttachmentStoreOp::DontCare,
            stencil_load_op: Some(AttachmentLoadOp::DontCare),
            stencil_store_op: Some(AttachmentStoreOp::DontCare),
            initial_layout: ImageLayout::Undefined,
            final_layout: ImageLayout::DepthStencilAttachmentOptimal,
            ..Default::default()
        };

        let color_attachment_resolve = AttachmentDescription {
            format: image_format,
            samples: SampleCount::Sample1,
            load_op: AttachmentLoadOp::DontCare,
            store_op: AttachmentStoreOp::Store,
            stencil_load_op: Some(AttachmentLoadOp::DontCare),
            stencil_store_op: Some(AttachmentStoreOp::DontCare),
            initial_layout: ImageLayout::Undefined,
            final_layout: if samples != SampleCount::Sample1 {
                ImageLayout::PresentSrc
            } else {
                ImageLayout::ColorAttachmentOptimal
            },
            ..Default::default()
        };

        let mut attachments = vec![color_attachment, depth_attachment];

        let color_attachment_reference = AttachmentReference {
            attachment: 0,
            layout: ImageLayout::ColorAttachmentOptimal,
            ..Default::default()
        };

        let depth_attachment_reference = AttachmentReference {
            attachment: 1,
            layout: ImageLayout::DepthStencilAttachmentOptimal,
            ..Default::default()
        };

        let color_attachments = vec![Some(color_attachment_reference)];

        let color_attachment_resolve_reference = AttachmentReference {
            attachment: 2,
            layout: ImageLayout::ColorAttachmentOptimal,
            ..Default::default()
        };

        let mut resolve_attachments = vec![];

        if samples != SampleCount::Sample1 {
            attachments.push(color_attachment_resolve);
            resolve_attachments.push(Some(color_attachment_resolve_reference))
        }

        let mut subpass = SubpassDescription {
            color_attachments: color_attachments,
            depth_stencil_attachment: Some(depth_attachment_reference),
            ..Default::default()
        };

        if samples != SampleCount::Sample1 {
            subpass.color_resolve_attachments = resolve_attachments;
        }

        let subpasses = vec![subpass];

        let dependency = SubpassDependency {
            src_subpass: None,
            dst_subpass: Some(0),
            src_stages: PipelineStages::COLOR_ATTACHMENT_OUTPUT
                | PipelineStages::EARLY_FRAGMENT_TESTS,
            src_access: AccessFlags::empty(),
            dst_stages: PipelineStages::COLOR_ATTACHMENT_OUTPUT
                | PipelineStages::EARLY_FRAGMENT_TESTS,
            dst_access: AccessFlags::COLOR_ATTACHMENT_WRITE
                | AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            ..Default::default()
        };

        let dependencies = vec![dependency];

        let render_pass_create_info = RenderPassCreateInfo {
            attachments: attachments,
            subpasses: subpasses,
            dependencies: dependencies,
            ..Default::default()
        };

        let render_pass = match RenderPass::new(device.clone(), render_pass_create_info) {
            Ok(render_pass) => render_pass,
            Err(e) => {
                log!("failed to crate render pass: {:?}", e);
                return Err(CrystalError::CannotCreateRenderPass);
            }
        };

        Ok(render_pass)
    }

    pub(crate) fn new(
        render_pass: Arc<RenderPass>,
        memory_allocator: Arc<StandardMemoryAllocator>,
        image_views: &[Arc<ImageView>],
        extent: [u32; 2],
    ) -> CrystalResult<Self> {
        let device = render_pass.device();
        let mut framebuffers = vec![];

        let samples = render_pass.attachments()[0].samples;
        let image_format = render_pass.attachments()[0].format;

        for image_view in image_views {
            let image_view = image_view.clone();

            let tiling = ImageTiling::Optimal;
            let depth_format = find_depth_format(
                device.clone(),
                tiling,
                FormatFeatures::DEPTH_STENCIL_ATTACHMENT,
            );

            let depth_image = Image::new(
                memory_allocator.clone(),
                extent,
                samples,
                depth_format,
                tiling,
                ImageAspects::DEPTH,
                ImageUsage::DEPTH_STENCIL_ATTACHMENT,
                false,
                1.,
            )?;

            let color_image = Image::new(
                memory_allocator.clone(),
                extent,
                samples,
                image_format,
                ImageTiling::Optimal,
                ImageAspects::COLOR,
                ImageUsage::TRANSIENT_ATTACHMENT | ImageUsage::COLOR_ATTACHMENT,
                false,
                1.,
            )?;

            let attachments = if samples != SampleCount::Sample1 {
                vec![color_image.image_view, depth_image.image_view, image_view]
            } else {
                vec![image_view, depth_image.image_view]
            };

            let create_info = FramebufferCreateInfo {
                attachments,
                extent,
                layers: 1,
                ..Default::default()
            };

            let framebuffer = match Framebuffer::new(render_pass.clone(), create_info) {
                Ok(framebuffer) => framebuffer,
                Err(e) => {
                    log!("cannot create framebuffer: {:?}", e);
                    return Err(CrystalError::CannotCreateRenderTarget);
                }
            };

            framebuffers.push(framebuffer);
        }

        Ok(Self {
            device: device.clone(),
            render_pass,
            framebuffers,
            frames_in_flight: image_views.len(),
            current_frame: 0,
            extent,
            samples,
        })
    }
}
