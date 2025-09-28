use crystal_api::{
    bitflags::BufferFlags,
    debug::{LoggingLevel, set_internal_logging_level},
    errors::GraphicsResult,
    *,
};

use std::{
    f32::consts::PI,
    fs::File,
    io::{BufRead, BufReader, Read},
    mem::offset_of,
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

const DISTANCE: f32 = 5.;

type Vec3 = [f32; 3];
type Vec2 = [f32; 2];

/// Index type
pub type Index = u16;

/// Textured vertex struct
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct VertexTexture {
    pos: Vec3,
    uv: Vec2,
}

impl AttributeDescriptor for VertexTexture {
    fn get_attributes() -> &'static [Attribute] {
        &[
            Attribute {
                size: size_of::<Vec3>(),
                offset: offset_of!(Self, pos),
            },
            Attribute {
                size: size_of::<Vec2>(),
                offset: offset_of!(Self, uv),
            },
        ]
    }
}

fn mesh_from_obj_buffer<T>(buffer: BufReader<T>) -> GraphicsResult<Mesh<VertexTexture, Index>>
where
    BufReader<T>: BufRead,
{
    let mut vertices = vec![];
    let mut indices = vec![];
    let mut uvs: Vec<[f32; 2]> = vec![];

    for line in buffer.lines() {
        let line = match line {
            Ok(line) => line,
            Err(_) => continue,
        };

        let splitted: Vec<&str> = line.split_whitespace().collect();

        if splitted.is_empty() || splitted[0].starts_with('#') {
            continue;
        }

        if splitted.len() >= 3 {
            match splitted[0] {
                "vt" => uvs.push([
                    splitted[1].parse::<f32>().unwrap(),
                    splitted[2].parse::<f32>().unwrap(),
                ]),
                "v" => vertices.push(VertexTexture {
                    pos: [
                        splitted[1].parse().unwrap(),
                        splitted[2].parse().unwrap(),
                        splitted[3].parse().unwrap(),
                    ],
                    uv: [0., 0.],
                }),
                "f" => {
                    let mut local_indices = vec![];

                    for &data in &splitted[1..] {
                        if data.starts_with('#') {
                            break;
                        }
                        let splitted: Vec<&str> = data.split('/').collect();

                        let idx: i32 = splitted[0].parse().unwrap();
                        let idx: Index = if idx >= 0 {
                            (idx - 1) as Index
                        } else {
                            (idx + vertices.len() as i32) as Index
                        };

                        if !splitted[1].is_empty() {
                            let uv: i32 = splitted[1].parse().unwrap();
                            let uv: usize = if uv >= 0 {
                                (uv - 1) as usize
                            } else {
                                (uv + uvs.len() as i32) as usize
                            };
                            vertices[idx as usize].uv = [uvs[uv][0], -uvs[uv][1]];
                        }

                        local_indices.push(idx);
                    }

                    if local_indices.len() == 3 {
                        indices.append(&mut local_indices);
                    } else if local_indices.len() == 4 {
                        indices.push(local_indices[0]);
                        indices.push(local_indices[1]);
                        indices.push(local_indices[2]);

                        indices.push(local_indices[0]);
                        indices.push(local_indices[2]);
                        indices.push(local_indices[3]);
                    }
                }

                _ => (),
            }
        }
    }

    Ok(Mesh { vertices, indices })
}

struct State {
    delta_time_sum: Duration,
    min_delta_time: Duration,
    max_delta_time: Duration,
    current_frame: usize,
    now: Instant,
    startup: Instant,
}

#[derive(Clone, Copy)]
pub struct Camera {
    pub view: glam::Mat4,
    pub proj: glam::Mat4,
}

impl Camera {
    fn calc_eye_matrix(&self) -> glam::Mat4 {
        self.proj * self.view
    }
}

#[repr(C, align(16))]
#[derive(Clone)]
struct Uniform {
    eye: glam::Mat4,
    time: f32,
}

struct ContextGraphics {
    camera: Camera,
    camera_in: Camera,

    uniform: Buffer<Uniform>,
    uniform_in: Buffer<Uniform>,

    transform: Buffer<glam::Mat4>,

    objects: Vec<Arc<Object>>,

    device: Device,

    state: State,
}

impl ContextGraphics {
    fn call_render(&mut self, window: &Window) {
        let now = self.state.startup.elapsed().as_secs_f32();

        let ubo = Uniform {
            eye: self.camera.calc_eye_matrix(),
            time: now,
        };

        self.uniform[0] = ubo;

        let ubo = Uniform {
            eye: self.camera_in.calc_eye_matrix(),
            time: now,
        };

        self.uniform_in[0] = ubo;

        self.transform[0] = glam::Mat4::from_scale_rotation_translation(
            glam::Vec3::new(0.03, 0.03, 0.03),
            glam::Quat::from_rotation_y(PI / 2. * now),
            glam::Vec3::ZERO,
        );

        self.transform[1] = glam::Mat4::from_scale_rotation_translation(
            glam::Vec3::from_array([0.5; 3]),
            glam::Quat::from_rotation_y(PI * 2. * now),
            glam::Vec3::ZERO,
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

        let binary_result1 = glsl2spirv!(
            "examples/shaders/render-target.vert",
            shaderc::ShaderKind::Vertex
        );
        let binary_result2 = glsl2spirv!(
            "examples/shaders/render-target.frag",
            shaderc::ShaderKind::Fragment
        );
        let binary_result3 = glsl2spirv!(
            "examples/shaders/textured.vert",
            shaderc::ShaderKind::Vertex
        );
        let binary_result4 = glsl2spirv!(
            "examples/shaders/textured.frag",
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

        let (render_target_sphere, texture_render) =
            render_target.inherit([1024, 1024], 1., 2).unwrap();

        let layout = device.create_layout(true, 1, 1, 2, 1).unwrap();

        let uniform = device
            .create_buffer(1, BufferFlags::UNIFORM | BufferFlags::SYNCED)
            .unwrap();
        let uniform_in = device
            .create_buffer(1, BufferFlags::UNIFORM | BufferFlags::SYNCED)
            .unwrap();
        let transform = device.create_buffer(2, BufferFlags::SYNCED).unwrap();

        layout.add_buffer(0, &uniform).unwrap();
        layout.add_buffer(1, &uniform_in).unwrap();
        layout.add_buffer(0, &transform).unwrap();

        let pipeline_render = layout
            .create_graphics_pipeline::<VertexTexture>(&render_target_sphere, &shaders_obj)
            .unwrap();

        let pipeline_textured = layout
            .create_graphics_pipeline::<VertexTexture>(&render_target, &shaders_textured)
            .unwrap();

        let mishka_mesh = mesh_from_obj_buffer(BufReader::new(
            File::open("examples/resources/mishka/owo.obj").unwrap(),
        ))
        .unwrap();

        let sphere_mesh = mesh_from_obj_buffer(BufReader::new(
            File::open("examples/resources/objects/uv-map-sphere.obj").unwrap(),
        ))
        .unwrap();

        let mishka_mesh_buffer = device.create_buffer_mesh(&mishka_mesh).unwrap();
        let sphere_mesh_buffer = device.create_buffer_mesh(&sphere_mesh).unwrap();

        let mishka_object =
            Arc::new(pipeline_render.create_object_with_mesh(1, &mishka_mesh_buffer));

        let texture_sampler = device
            .create_sampler_set(&[(0, &texture_render)], &[&layout])
            .unwrap();

        let sphere_object = Arc::new(pipeline_textured.create_object_with_mesh_sampled(
            0,
            &sphere_mesh_buffer,
            texture_sampler,
        ));

        const DISTANCE_FROM_OBJECTS: f32 = DISTANCE + 1.;

        let camera = Camera {
            proj: glam::Mat4::perspective_lh(
                PI / 4.,
                self.settings.width as f32 / self.settings.height as f32,
                0.1,
                100.,
            ),
            view: glam::Mat4::look_at_lh(
                glam::Vec3::new(1., 0., 1.) * DISTANCE_FROM_OBJECTS,
                glam::Vec3::ZERO,
                glam::Vec3::new(0., -1., 0.),
            ),
        };

        let camera_in = Camera {
            proj: glam::Mat4::perspective_lh(PI / 4., 1., 0.1, 100.),
            view: glam::Mat4::look_at_lh(
                glam::Vec3::new(
                    DISTANCE_FROM_OBJECTS,
                    DISTANCE_FROM_OBJECTS,
                    DISTANCE_FROM_OBJECTS,
                ),
                glam::Vec3::ZERO,
                glam::Vec3::new(0., -1., 0.),
            ),
        };

        window.request_redraw();
        println!("[end init]");

        self.window = Some(window);

        self.context = Some(ContextGraphics {
            camera,
            camera_in,
            uniform,
            uniform_in,
            transform,
            objects: vec![sphere_object, mishka_object],
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
                        _ => (),
                    }
                }
            }
            WindowEvent::Resized(size) => {
                context.camera.proj = glam::Mat4::perspective_lh(
                    PI / 4.,
                    size.width as f32 / size.height as f32,
                    0.1,
                    100.,
                );
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
