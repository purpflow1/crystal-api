use crystal_api::{
    debug::{LoggingLevel, set_internal_logging_level},
    errors::GraphicsResult,
    object::Object,
    *,
};

use std::{
    f32::consts::PI,
    fs::File,
    io::{BufReader, Read},
    sync::Arc,
    time::{Duration, Instant},
};

use mesh::{Mesh, VertexTexture};
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::WindowEvent,
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};

const DISTANCE: f32 = 5.;

struct State {
    delta_time_sum: Duration,
    current_frame: usize,
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

struct Scene {
    camera: Camera,

    uniform: Option<Arc<dyn Buffer>>,
    transforms: Option<Arc<dyn Buffer>>,

    objects: Vec<Arc<Object>>,
}

struct Context {
    window: Option<Window>,
    graphics: Option<Arc<dyn GraphicsApi>>,
    render_target_cube: Option<Arc<dyn RenderTarget>>,

    settings: GraphicsApiInitSettings,
    scene: Scene,

    state: State,
}

impl Context {
    pub fn new(settings: GraphicsApiInitSettings) -> GraphicsResult<Self> {
        let width = settings.width;
        let height = settings.height;

        const DISTANCE_FROM_OBJECTS: f32 = DISTANCE + 1.;

        Ok(Self {
            window: None,
            graphics: None,
            render_target_cube: None,

            settings,
            scene: Scene {
                camera: Camera {
                    proj: glam::Mat4::perspective_lh(
                        PI / 4.,
                        width as f32 / height as f32,
                        0.1,
                        100.,
                    ),
                    view: glam::Mat4::look_at_lh(
                        glam::Vec3::new(
                            DISTANCE_FROM_OBJECTS,
                            DISTANCE_FROM_OBJECTS,
                            DISTANCE_FROM_OBJECTS,
                        ),
                        glam::Vec3::ZERO,
                        glam::Vec3::new(0., -1., 0.),
                    ),
                },

                uniform: None,
                transforms: None,

                objects: vec![],
            },

            state: State {
                delta_time_sum: Duration::ZERO,
                current_frame: 0,
                startup: std::time::Instant::now(),
            },
        })
    }
}

impl ApplicationHandler for Context {
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

        let graphics = init_api_instance_with_presentation(&self.settings, &window)
            .expect("cannot create entry");

        println!("compiling GLSL shaders...");
        let file_name1 = "examples/render-target/shaders/desc.vert";
        let file_name2 = "examples/render-target/shaders/desc.frag";
        let file_name3 = "examples/render-target/shaders/textured.frag";
        let mut source1 = String::new();
        let mut source2 = String::new();
        let mut source3 = String::new();
        let mut reader = BufReader::new(File::open(file_name1).unwrap());
        reader.read_to_string(&mut source1).unwrap();
        let mut reader = BufReader::new(File::open(file_name2).unwrap());
        reader.read_to_string(&mut source2).unwrap();
        let mut reader = BufReader::new(File::open(file_name3).unwrap());
        reader.read_to_string(&mut source3).unwrap();

        let compiler = shaderc::Compiler::new().unwrap();
        let mut options = shaderc::CompileOptions::new().unwrap();
        options.add_macro_definition("EP", Some("main"));
        let binary_result1 = compiler
            .compile_into_spirv(
                source1.as_str(),
                shaderc::ShaderKind::Vertex,
                file_name1,
                "main",
                Some(&options),
            )
            .unwrap();
        let binary_result2 = compiler
            .compile_into_spirv(
                source2.as_str(),
                shaderc::ShaderKind::Fragment,
                file_name2,
                "main",
                Some(&options),
            )
            .unwrap();
        let binary_result3 = compiler
            .compile_into_spirv(
                source3.as_str(),
                shaderc::ShaderKind::Fragment,
                file_name3,
                "main",
                Some(&options),
            )
            .unwrap();

        let shaders_obj = [
            Shader::from_bytes(binary_result1.as_binary_u8(), ShaderStage::Vertex).unwrap(),
            Shader::from_bytes(binary_result2.as_binary_u8(), ShaderStage::Fragment).unwrap(),
        ];

        let shaders_textured = [
            Shader::from_bytes(binary_result1.as_binary_u8(), ShaderStage::Vertex).unwrap(),
            Shader::from_bytes(binary_result3.as_binary_u8(), ShaderStage::Fragment).unwrap(),
        ];

        let render_target = graphics.get_presentation_render_target().unwrap();

        let (render_target_cube, texture_render) = render_target
            .create_render_target([1024, 1024], 1., 2)
            .unwrap();

        let layout = graphics.create_layout(true, 1, 1, 1, 1).unwrap();

        let uniform = graphics
            .create_buffer(size_of::<Uniform>() as u64, true, false, true)
            .unwrap();
        let transform = graphics
            .create_buffer(size_of::<glam::Mat4>() as u64 * 2, false, false, true)
            .unwrap();

        layout.add_buffer(0, uniform.clone()).unwrap();
        layout.add_buffer(0, transform.clone()).unwrap();

        self.scene.uniform = Some(uniform);
        self.scene.transforms = Some(transform);

        let pipeline_render = layout
            .clone()
            .create_graphics_pipeline(
                render_target_cube.clone(),
                &shaders_obj,
                &VertexTexture::get_attributes(),
            )
            .unwrap();

        let pipeline_textured = layout
            .clone()
            .create_graphics_pipeline(
                render_target.clone(),
                &shaders_textured,
                &VertexTexture::get_attributes(),
            )
            .unwrap();

        let mishka_mesh = Arc::new(
            Mesh::from_buffer(BufReader::new(
                File::open("examples/render-target/resources/mishka/owo.obj").unwrap(),
            ))
            .unwrap(),
        );

        let cube_mesh = Arc::new(
            Mesh::from_buffer(BufReader::new(
                File::open("examples/render-target/resources/objects/cube.obj").unwrap(),
            ))
            .unwrap(),
        );

        let mishka_mesh_buffer = graphics.create_buffer_mesh(mishka_mesh).unwrap();
        let cube_mesh_buffer = graphics.create_buffer_mesh(cube_mesh).unwrap();

        let mishka_object =
            Object::with_mesh(1, pipeline_render.clone(), mishka_mesh_buffer.clone());

        let texture_sampler = graphics
            .create_sampler_set(&[(0, texture_render)], &[layout])
            .unwrap();

        let cube_object = Object::with_mesh_sampled(
            0,
            pipeline_textured.clone(),
            cube_mesh_buffer.clone(),
            texture_sampler,
        );

        self.scene.objects.push(cube_object.clone());
        self.scene.objects.push(mishka_object.clone());

        self.graphics = Some(graphics);
        self.window = Some(window);
        self.render_target_cube = Some(render_target_cube);

        println!("[end init]");
    }

    fn about_to_wait(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        let ubo = Uniform {
            eye: self.scene.camera.calc_eye_matrix(),
            time: self.state.startup.elapsed().as_secs_f32(),
        };

        self.scene
            .uniform
            .as_ref()
            .unwrap()
            .get_memory_full()
            .copy_from_slice(vec![ubo].as_bytes());

        let transforms = vec![
            glam::Mat4::from_scale_rotation_translation(
                glam::Vec3::ONE,
                glam::Quat::from_rotation_y(PI / 2. * self.state.delta_time_sum.as_secs_f32()),
                glam::Vec3::ZERO,
            ),
            glam::Mat4::from_scale_rotation_translation(
                glam::Vec3::from_array([0.5; 3]),
                glam::Quat::from_rotation_y(PI * 2. * self.state.delta_time_sum.as_secs_f32()),
                glam::Vec3::ZERO,
            ),
        ];

        self.scene
            .transforms
            .as_ref()
            .unwrap()
            .get_memory_full()
            .copy_from_slice(transforms.as_bytes());

        let graphics = self.graphics.clone().unwrap();

        self.state.delta_time_sum += graphics.get_delta_time();
        self.state.current_frame += 1;

        self.graphics
            .as_ref()
            .unwrap()
            .dispatch_and_present(&self.scene.objects)
            .unwrap();
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
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
            } => {}
            WindowEvent::Resized(size) => {
                self.scene.camera.proj = glam::Mat4::perspective_lh(
                    PI / 4.,
                    size.width as f32 / size.height as f32,
                    0.1,
                    100.,
                );

                self.graphics
                    .as_ref()
                    .unwrap()
                    .resize_resources(size.width, size.height)
                    .unwrap();
            }
            WindowEvent::RedrawRequested => {
                let window = self.window.as_ref().unwrap();
                window.request_redraw();
            }
            _ => {}
        }
    }

    fn exiting(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        self.graphics = None;
    }
}

fn main() -> GraphicsResult<()> {
    set_internal_logging_level(LoggingLevel::Console);

    let settings = GraphicsApiInitSettings::default()
        .msaa_samples(4)
        .width(1000)
        .height(700);

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut context = Context::new(settings)?;
    event_loop
        .run_app(&mut context)
        .expect("cannot run event loop");

    Ok(())
}
