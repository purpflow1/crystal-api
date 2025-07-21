use crystal_api::{errors::CrystalResult, object::Object, vulkan::VulkanEntry, *};
use sysinfo::{Pid, PidExt, ProcessExt, System, SystemExt};

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

const DEBUG_OUTPUT: bool = true;
const OBJECT_DIMENTION: usize = 10;
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
struct Vec3 {
    x: f32,
    y: f32,
    z: f32,
}

#[repr(C, align(16))]
#[derive(Clone)]
struct Particle {
    pos: Vec3,
    vel: Vec3,
}

impl std::fmt::Debug for Particle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "({:.1}, {:.1}, {:.1}) ({:.1}, {:.1}, {:.1})",
            self.pos.x, self.pos.y, self.pos.z, self.vel.x, self.vel.y, self.vel.z
        ))
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
    pub fn new(path: &Path) -> CrystalResult<Self> {
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

    compute_buffer_in: Option<Arc<dyn Buffer>>,
    compute_buffer_out: Option<Arc<dyn Buffer>>,

    objects: Vec<Arc<Object>>,
}

struct Context {
    window: Option<Window>,
    graphics: Option<Arc<VulkanEntry>>,

    obj_compute: Option<Arc<Object>>,

    settings: GraphicsApiInitSettings,
    scene: Scene,

    state: State,

    system: System,
}

impl Context {
    pub fn new(settings: GraphicsApiInitSettings) -> CrystalResult<Self> {
        let width = settings.width;
        let height = settings.height;

        const DISTANCE_FROM_OBJECTS: f32 = (DISTANCE + 1.) * OBJECT_DIMENTION as f32;

        Ok(Self {
            window: None,
            graphics: None,
            obj_compute: None,
            system: System::new_all(),

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
                compute_buffer_in: None,
                compute_buffer_out: None,

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

        let graphics = VulkanEntry::with_presentation(&self.settings, &window)
            .expect("cannot create vulkan entry");

        let shaders_obj = [
            Shader::open("shaders/desc.vert.spv", ShaderStage::Vertex).unwrap(),
            Shader::open("shaders/desc.frag.spv", ShaderStage::Fragment).unwrap(),
        ];

        let shader_compute =
            Shader::open("shaders/particles.comp.spv", ShaderStage::Compute).unwrap();

        let render_target = graphics.get_viewport();

        let layout_obj = graphics
            .create_layout(true, 2, OBJECT_DIMENTION.pow(3), 1, 3)
            .unwrap();
        let layout_compute = graphics.create_layout(true, 0, 0, 1, 2).unwrap();
        // let layout_graph = graphics.create_layout(true, texture_num, sampler_num, uniform_num, storage_num)

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

        let compute_buffer_in = graphics
            .create_buffer(size_of::<Particle>() as u64 * 256, false, false, true)
            .unwrap();
        let compute_buffer_out = graphics
            .create_buffer(size_of::<Particle>() as u64 * 256, false, true, true)
            .unwrap();

        layout_obj.add_buffer(0, uniform.clone()).unwrap();
        layout_obj.add_buffer(0, transform.clone()).unwrap();
        layout_obj.add_buffer(1, light.clone()).unwrap();
        layout_obj.add_buffer(2, light_info.clone()).unwrap();
        layout_compute.add_buffer(0, uniform.clone()).unwrap();
        layout_compute
            .add_buffer(0, compute_buffer_in.clone())
            .unwrap();
        layout_compute
            .add_buffer(1, compute_buffer_out.clone())
            .unwrap();

        self.scene.uniform = Some(uniform);
        self.scene.transforms = Some(transform);
        self.scene.light = Some(light);
        self.scene.light_info = Some(light_info);
        self.scene.compute_buffer_in = Some(compute_buffer_in);
        self.scene.compute_buffer_out = Some(compute_buffer_out);

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
        self.scene
            .compute_buffer_in
            .as_ref()
            .unwrap()
            .get_memory_full()
            .copy_from_slice(
                &(1..=256)
                    .map(|n| Particle {
                        pos: Vec3 {
                            x: 3. / n as f32 - 1.5,
                            y: 4.,
                            z: 3. / n as f32 - 1.5,
                        },
                        vel: Vec3 {
                            x: 0.,
                            y: -1.,
                            z: 0.,
                        },
                    })
                    .collect::<Vec<Particle>>()
                    .as_bytes(),
            );

        let pipeline_render = layout_obj
            .clone()
            .create_graphics_pipeline(
                render_target.clone(),
                &shaders_obj,
                &VertexTexture::get_attributes(),
            )
            .unwrap();

        let pipeline_compute = layout_compute
            .clone()
            .create_compute_pipeline(&shader_compute)
            .unwrap();

        let obj_compute = Object::new_compute(pipeline_compute, [1, 1, 1]);

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
        self.obj_compute = Some(obj_compute);
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

        if DEBUG_OUTPUT && self.state.delta_time_sum.as_secs_f64() >= 1. {
            let pid = Pid::from_u32(std::process::id());

            self.system.refresh_process(pid);

            let process = self.system.process(pid).unwrap();

            let gpu_debug = graphics.get_debug_data();

            macro_rules! as_mb {
                ($kb:expr) => {
                    $kb as f32 / 1024. / 1024.
                };
            }

            println!("[DEBUG]");
            println!("FPS: {}", self.state.current_frame);
            println!("GPU mem:   {:.1} MB", as_mb!(gpu_debug.used_memory));
            println!("RAM usage: {:.1} MB", process.memory() as f32 / 1024.);
            println!("CPU usage: {:.1}%", process.cpu_usage());
            println!();

            let mem = self
                .scene
                .compute_buffer_out
                .as_ref()
                .unwrap()
                .get_memory(0..8 * size_of::<f32>());

            println!("{:?}\n", unsafe {
                std::slice::from_raw_parts(mem.as_ptr() as *const Particle, 2)
            });

            self.state.delta_time_sum = Duration::ZERO;
            self.state.current_frame = 0;
        }

        self.graphics
            .clone()
            .unwrap()
            .dispatch_any(
                &self
                    .scene
                    .objects
                    .clone()
                    .into_iter()
                    .chain([self.obj_compute.clone().unwrap()].into_iter())
                    .collect::<Vec<Arc<Object>>>(),
            )
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
                    .recreate_resources(size.width, size.height)
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

fn main() -> CrystalResult<()> {
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
