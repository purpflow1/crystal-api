use crystal_api::{
    bitflags::BufferFlags,
    debug::{LoggingLevel, set_internal_logging_level},
    errors::GraphicsResult,
    *,
};

use std::{
    f32::consts::PI,
    fs::File,
    io::{BufReader, Read},
    sync::Arc,
    time::{Duration, Instant},
};

use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::WindowEvent,
    event_loop::{ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::Window,
};

type Vec3 = [f32; 3];
type Vec2 = [f32; 2];

/// Index type
pub type Index = u16;

/// Textured vertex struct
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct VertexTexture(Vec3, Vec2);

impl AttributeDescriptor for VertexTexture {
    fn get_attributes() -> &'static [Attribute] {
        &[
            Attribute {
                size: size_of::<Vec3>(),
                offset: 0,
            },
            Attribute {
                size: size_of::<Vec2>(),
                offset: size_of::<Vec3>(),
            },
        ]
    }
}

struct State {
    delta_time_sum: Duration,
    min_delta_time: Duration,
    max_delta_time: Duration,
    current_frame: usize,
    now: Instant,
    startup: Instant,
}

struct ContextGraphics {
    uniform: Buffer<glam::Vec2>,
    transform: Buffer<glam::Mat4>,

    objects: Vec<Arc<Object>>,

    device: Device,

    state: State,
}

impl ContextGraphics {
    fn call_render(&mut self, window: &Window) {
        let now = self.state.startup.elapsed().as_secs_f32();

        let aspect_ratio = window.inner_size().width as f32 / window.inner_size().height as f32;
        self.uniform[0] = glam::Vec2::new(
            window.inner_size().width as f32,
            window.inner_size().height as f32,
        );

        let camera = glam::Mat4::perspective_lh(PI / 3., aspect_ratio, 0.1, 100.)
            * glam::Mat4::look_at_lh(
                glam::Vec3::new(0., 0., -1.),
                glam::Vec3::ZERO,
                glam::Vec3::new(0., 1., 0.),
            );

        self.transform[0] = camera
            * glam::Mat4::from_scale_rotation_translation(
                glam::Vec3::new(0.8, 0.8, 0.8),
                glam::Quat::from_rotation_y(now * PI / 10.)
                    * glam::Quat::from_rotation_z(now * PI / 10.),
                glam::Vec3::new(0., 0., 1.),
            );

        self.device.dispatch_and_present(&self.objects).unwrap();

        let now = Instant::now();
        let delta = now - self.state.now;
        self.state.now = now;

        self.state.delta_time_sum += delta;

        if self.state.min_delta_time > delta {
            self.state.min_delta_time = delta
        };
        if self.state.max_delta_time < delta {
            self.state.max_delta_time = delta
        };

        if self.state.delta_time_sum > Duration::from_secs(1) {
            window.set_title(
                format!(
                    "FPS: [ avg: {} min: {} max: {} ]",
                    self.state.current_frame,
                    (1. / self.state.max_delta_time.as_secs_f32()) as u32,
                    (1. / self.state.min_delta_time.as_secs_f32()) as u32
                )
                .as_str(),
            );

            self.state.delta_time_sum = Duration::ZERO;
            self.state.min_delta_time = Duration::MAX;
            self.state.max_delta_time = Duration::ZERO;
            self.state.current_frame = 0;
        }

        self.state.current_frame += 1;
    }
}

struct ContextWindow {
    settings: GraphicsApiInitSettings,
    context: Option<ContextGraphics>,
    window: Option<Window>,
}

impl ContextWindow {
    pub fn new(settings: GraphicsApiInitSettings) -> GraphicsResult<Self> {
        Ok(Self {
            settings,
            context: None,
            window: None,
        })
    }
}

impl ApplicationHandler for ContextWindow {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let window = {
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_inner_size(LogicalSize::new(
                            self.settings.width,
                            self.settings.height,
                        ))
                        .with_min_inner_size(LogicalSize::new(
                            self.settings.width / 4,
                            self.settings.height / 4,
                        )),
                )
                .unwrap()
        };

        let device = Device::graphics(&self.settings, &window).expect("cannot create entry");

        let compiler = shaderc::Compiler::new().unwrap();

        macro_rules! glsl2spirv {
            ($filename:expr, $shaderkind:expr) => {{
                const FILENAME: &str = $filename;
                let mut source = String::new();
                let mut reader = BufReader::new(File::open(FILENAME).unwrap());
                reader.read_to_string(&mut source).unwrap();
                compiler
                    .compile_into_spirv(source.as_str(), $shaderkind, FILENAME, "main", None)
                    .unwrap()
            }};
        }

        let binary_result1 =
            glsl2spirv!("examples/shaders/plain.vert", shaderc::ShaderKind::Vertex);
        let binary_result2 =
            glsl2spirv!("examples/shaders/plain.frag", shaderc::ShaderKind::Fragment);
        let binary_result3 = glsl2spirv!(
            "examples/shaders/post-process.vert",
            shaderc::ShaderKind::Vertex
        );
        let binary_result4 = glsl2spirv!(
            "examples/shaders/post-process.frag",
            shaderc::ShaderKind::Fragment
        );

        let shaders_obj = [
            Shader::from_bytes(binary_result1.as_binary_u8(), ShaderStage::Vertex).unwrap(),
            Shader::from_bytes(binary_result2.as_binary_u8(), ShaderStage::Fragment).unwrap(),
        ];

        let shaders_textured = [
            Shader::from_bytes(binary_result3.as_binary_u8(), ShaderStage::Vertex).unwrap(),
            Shader::from_bytes(binary_result4.as_binary_u8(), ShaderStage::Fragment).unwrap(),
        ];

        let render_target = device.get_presentation_render_target().unwrap();

        let (render_target_fullscreen, texture_render) =
            render_target.inherit([1024, 1024], 1., 2).unwrap();

        let layout = device.create_layout(true, 1, 1, 1, 1).unwrap();

        let transform = device.create_buffer(1, BufferFlags::SYNCED).unwrap();
        let uniform = device
            .create_buffer(1, BufferFlags::UNIFORM | BufferFlags::SYNCED)
            .unwrap();

        layout.add_buffer(0, &transform).unwrap();
        layout.add_buffer(0, &uniform).unwrap();

        let pipeline_render = layout
            .create_graphics_pipeline::<VertexTexture>(&render_target_fullscreen, &shaders_obj)
            .unwrap();

        let pipeline_textured = layout
            .create_graphics_pipeline::<VertexTexture>(&render_target, &shaders_textured)
            .unwrap();

        let cube_mesh = Mesh {
            vertices: vec![
                // bottom
                VertexTexture([0.5, -0.5, 0.5], [0.0, 0.0]),
                VertexTexture([0.5, -0.5, -0.5], [0.5, 0.0]),
                VertexTexture([-0.5, -0.5, 0.5], [0.0, 0.5]),
                VertexTexture([-0.5, -0.5, -0.5], [0.5, 0.5]),
                // top
                VertexTexture([0.5, 0.5, 0.5], [0.5, 0.5]),
                VertexTexture([0.5, 0.5, -0.5], [1., 0.5]),
                VertexTexture([-0.5, 0.5, 0.5], [0.5, 1.]),
                VertexTexture([-0.5, 0.5, -0.5], [1., 1.]),
            ],
            indices: vec![
                0, 2, 1, 1, 2, 3, // bottom
                4, 5, 6, 5, 7, 6, // top
                0, 4, 2, 2, 4, 6, // front
                1, 3, 5, 3, 7, 5, // back
                0, 1, 4, 1, 5, 4, // right
                2, 6, 3, 3, 6, 7, // left
            ],
        };

        let fullscreen_mesh = Mesh {
            vertices: vec![
                VertexTexture([-1., -1., 0.], [0., 0.]),
                VertexTexture([1., -1., 0.], [1., 0.]),
                VertexTexture([1., 1., 0.], [1., 1.]),
                VertexTexture([-1., 1., 0.], [0., 1.]),
            ],
            indices: vec![0, 2, 1, 3, 2, 0],
        };

        let cube_mesh_buffer = device.create_buffer_mesh(&cube_mesh).unwrap();
        let fullscreen_mesh_buffer = device.create_buffer_mesh(&fullscreen_mesh).unwrap();

        let cube_object = Arc::new(pipeline_render.create_object_with_mesh(1, &cube_mesh_buffer));

        let texture_sampler = device
            .create_sampler_set(&[(0, &texture_render)], &[&layout])
            .unwrap();

        let fullscreen_object = Arc::new(pipeline_textured.create_object_with_mesh_sampled(
            0,
            &fullscreen_mesh_buffer,
            texture_sampler,
        ));

        window.request_redraw();
        println!("[end init]");

        self.window = Some(window);

        self.context = Some(ContextGraphics {
            transform,
            uniform,
            objects: vec![fullscreen_object, cube_object],
            device,
            state: State {
                delta_time_sum: Duration::ZERO,
                min_delta_time: Duration::MAX,
                max_delta_time: Duration::ZERO,
                current_frame: 0,
                startup: std::time::Instant::now(),
                now: std::time::Instant::now(),
            },
        });
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        let window = self.window.as_ref().unwrap();
        let context = self.context.as_mut().unwrap();

        match event {
            WindowEvent::CloseRequested => {
                println!("Stopping window context with close request");
                event_loop.exit();
            }
            #[allow(unused)]
            WindowEvent::KeyboardInput {
                device_id,
                event,
                is_synthetic,
            } => {
                if !event.repeat && event.state.is_pressed() {
                    match event.physical_key {
                        PhysicalKey::Code(KeyCode::Space) => {
                            println!("usage: {}", context.device.get_memory_usage_fmt())
                        }
                        PhysicalKey::Code(KeyCode::ArrowUp) => {}
                        _ => (),
                    }
                }
            }
            WindowEvent::Resized(_size) => {
                context.call_render(window);
                window.request_redraw();
            }
            WindowEvent::RedrawRequested => {
                context.call_render(window);
                window.request_redraw();
            }
            _ => {}
        }
    }

    fn exiting(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        // Wayland surface can be destroyed before vulkan resources removal
        self.context = None;
    }
}

fn main() -> GraphicsResult<()> {
    set_internal_logging_level(LoggingLevel::Console);

    let settings = GraphicsApiInitSettings::default()
        .msaa_samples(4)
        .vsync(false)
        .width(500)
        .height(300);

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut context = ContextWindow::new(settings)?;
    event_loop
        .run_app(&mut context)
        .expect("cannot run event loop");

    Ok(())
}
