use crystal_api::{errors::GraphicsResult, object::Object, *};

use std::{
    f32::consts::PI,
    fs::File,
    io::BufReader,
    path::Path,
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

const OBJECT_DIMENTION: usize = 8;
const DISTANCE: f32 = 2.;

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

#[repr(C, align(16))]
#[derive(Clone)]
struct Light(glam::Vec3);

pub struct Image2D {
    pub width: u32,
    pub height: u32,
    pub channels: u32,
    pub pixels: Vec<u8>,
}

impl Image2D {
    pub fn new(path: &Path) -> GraphicsResult<Self> {
        let file = File::open(path).unwrap();
        let decoder = png::Decoder::new(file);
        let mut reader = decoder.read_info().unwrap();
        let mut pixels = vec![0; reader.output_buffer_size()];
        let info = reader.next_frame(&mut pixels).unwrap();

        Ok(Self {
            width: info.width,
            height: info.height,
            channels: info.bit_depth as u32,
            pixels,
        })
    }
}

struct Scene {
    camera: Camera,

    light: Option<Arc<dyn Buffer>>,
    light_info: Option<Arc<dyn Buffer>>,

    uniform: Option<Arc<dyn Buffer>>,
    transforms: Option<Arc<dyn Buffer>>,

    objects: Vec<Arc<Object>>,
}

struct Context {
    window: Option<Window>,
    graphics: Option<Arc<dyn GraphicsApi>>,

    settings: GraphicsApiInitSettings,
    scene: Scene,

    state: State,
}

impl Context {
    pub fn new(settings: GraphicsApiInitSettings) -> GraphicsResult<Self> {
        let width = settings.width;
        let height = settings.height;

        const DISTANCE_FROM_OBJECTS: f32 = (DISTANCE + 1.) * OBJECT_DIMENTION as f32;

        Ok(Self {
            window: None,
            graphics: None,

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
                light: None,
                light_info: None,
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

        let shaders_obj = [
            Shader::open(
                "examples/array-load-test/shaders/desc.vert.spv",
                ShaderStage::Vertex,
            )
            .unwrap(),
            Shader::open(
                "examples/array-load-test/shaders/desc.frag.spv",
                ShaderStage::Fragment,
            )
            .unwrap(),
        ];

        let render_target = graphics.get_presentation_render_target();

        let layout_obj = graphics
            .create_layout(true, 2, OBJECT_DIMENTION.pow(3), 1, 3)
            .unwrap();

        let default_sampler = {
            let file = File::open("resources/textures/default.png").unwrap();
            let decoder = png::Decoder::new(file);
            let mut reader = decoder.read_info().unwrap();

            let size = reader.output_buffer_size();
            let buffer = graphics
                .create_buffer(size as u64 * 2, false, true, false)
                .unwrap();

            let info = reader.next_frame(buffer.get_memory(0..size)).unwrap();

            let texture = graphics
                .create_texture(
                    buffer,
                    [info.width, info.height, info.bit_depth as u32],
                    1.0,
                )
                .unwrap();
            graphics.create_sampler_set(&[(0, texture)]).unwrap()
        };

        let _test_sampler = {
            let file = File::open("resources/textures/test.png").unwrap();
            let decoder = png::Decoder::new(file);
            let mut reader = decoder.read_info().unwrap();

            let size = reader.output_buffer_size();
            let buffer = graphics
                .create_buffer(size as u64 * 2, false, true, false)
                .unwrap();

            let info = reader.next_frame(buffer.get_memory(0..size)).unwrap();

            let texture = graphics
                .create_texture(
                    buffer,
                    [info.width, info.height, info.bit_depth as u32],
                    1.0,
                )
                .unwrap();
            graphics.create_sampler_set(&[(0, texture)]).unwrap()
        };

        let uniform = graphics
            .create_buffer(size_of::<Uniform>() as u64, true, false, true)
            .unwrap();
        let transform = graphics
            .create_buffer(
                (size_of::<glam::Mat4>() * OBJECT_DIMENTION.pow(3)) as u64,
                false,
                false,
                true,
            )
            .unwrap();
        let light = graphics
            .create_buffer(size_of::<Light>() as u64 * 3, false, false, true)
            .unwrap();
        let light_info = graphics
            .create_buffer(size_of::<u32>() as u64, false, false, true)
            .unwrap();

        layout_obj.add_buffer(0, uniform.clone()).unwrap();
        layout_obj.add_buffer(0, transform.clone()).unwrap();
        layout_obj.add_buffer(1, light.clone()).unwrap();
        layout_obj.add_buffer(2, light_info.clone()).unwrap();

        self.scene.uniform = Some(uniform);
        self.scene.transforms = Some(transform);
        self.scene.light = Some(light);
        self.scene.light_info = Some(light_info);

        self.scene
            .light
            .as_ref()
            .unwrap()
            .get_memory_full()
            .copy_from_slice(
                vec![
                    Light(glam::Vec3 {
                        x: 1.,
                        y: 1.,
                        z: 1.,
                    }),
                    Light(glam::Vec3 {
                        x: 0.,
                        y: 3.,
                        z: 0.,
                    }),
                    Light(glam::Vec3 {
                        x: 0.,
                        y: 0.,
                        z: 0.,
                    }),
                ]
                .as_bytes(),
            );
        self.scene
            .light_info
            .as_ref()
            .unwrap()
            .get_memory_full()
            .copy_from_slice(&1u32.to_le_bytes());

        let pipeline_render = layout_obj
            .clone()
            .create_graphics_pipeline(
                render_target.clone(),
                &shaders_obj,
                &VertexTexture::get_attributes(),
            )
            .unwrap();

        let mesh = Arc::new(
            Mesh::from_buffer(BufReader::new(
                File::open("resources/mishka/owo.obj").unwrap(),
            ))
            .unwrap(),
        );

        let mesh_buffer = graphics.create_buffer_mesh(mesh).unwrap();

        let object = Object::with_mesh_sampled_array(
            pipeline_render.clone(),
            mesh_buffer.clone(),
            default_sampler.clone(),
            OBJECT_DIMENTION.pow(3) as u32,
        );

        self.scene.objects.push(object.clone());
        layout_obj
            .register_samplers(&[default_sampler, _test_sampler])
            .unwrap();

        self.graphics = Some(graphics);
        self.window = Some(window);
    }

    fn about_to_wait(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        let rotation_matrix = glam::Quat::from_mat4(&glam::Mat4::from_rotation_y(
            PI * 2. * self.state.delta_time_sum.as_secs_f32(),
        ));

        let mut transforms = Vec::with_capacity(OBJECT_DIMENTION.pow(3));

        (1..=OBJECT_DIMENTION).for_each(|i| {
            (1..=OBJECT_DIMENTION).for_each(|j| {
                (1..=OBJECT_DIMENTION).for_each(|k| {
                    let transform = glam::Mat4::from_scale_rotation_translation(
                        glam::Vec3::new(0.3, 0.3, 0.3),
                        rotation_matrix,
                        glam::Vec3::new(i as f32, j as f32, k as f32) * DISTANCE,
                    );

                    transforms.push(transform);
                })
            })
        });

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
            .dispatch_any(&self.scene.objects)
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
