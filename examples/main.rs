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

use images::Image2D;
use mesh::{Mesh, VertexTexture};
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::WindowEvent,
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};

const MAX_INSTANCE_NUM: usize = 3;

struct State {
    delta_time: Duration,
    delta_time_sum: Duration,
    current_frame: usize,
    now: Option<Instant>,
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

struct Scene {
    camera: Camera,

    light: Arc<GpuVec<[f32; 3]>>,
    light_info: Arc<GpuVec<[u32; 3]>>,

    uniform: Arc<GpuVec<Uniform>>,
    transforms: Arc<GpuVec<glam::Mat4>>,

    compute_buffer_in: Arc<GpuVec<Particle>>,
    compute_buffer_out: Arc<GpuVec<Particle>>,

    objects: Vec<Arc<Object>>,
}

struct Context {
    window: Option<Window>,
    graphics: Option<Arc<VulkanEntry>>,

    layout_obj: Option<Arc<dyn Layout>>,
    layout_compute: Option<Arc<dyn Layout>>,
    pipeline_compute: Option<Arc<dyn Pipeline>>,

    settings: GraphicsApiInitSettings,
    scene: Scene,

    state: State,

    system: System,
}

impl Context {
    pub fn new(settings: GraphicsApiInitSettings) -> CrystalResult<Self> {
        let width = settings.width;
        let height = settings.height;

        Ok(Self {
            window: None,
            graphics: None,
            layout_obj: None,
            layout_compute: None,
            pipeline_compute: None,
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
                        glam::Vec3::new(5., 5., 5.),
                        glam::Vec3::ZERO,
                        glam::Vec3::new(0., -1., 0.),
                    ),
                },

                light: GpuVec::with_len(60),
                light_info: GpuVec::with_len(3),
                transforms: GpuVec::with_len(MAX_INSTANCE_NUM),
                compute_buffer_in: GpuVec::with_len(256),
                compute_buffer_out: GpuVec::with_len_transfer(256),

                uniform: GpuVec::with_len(1),
                objects: vec![],
            },

            state: State {
                delta_time: Duration::ZERO,
                delta_time_sum: Duration::ZERO,
                current_frame: 0,
                now: None,
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

        let default_texture_image =
            Image2D::new(Path::new("resources/textures/default.png")).unwrap();
        let default_sampler = graphics
            .create_sampler(&default_texture_image, 1.0)
            .unwrap();

        let test_texture_image = Image2D::new(Path::new("resources/textures/test.png")).unwrap();
        let test_sampler = graphics.create_sampler(&test_texture_image, 1.0).unwrap();

        let layout_obj = graphics.create_layout(true, 2, 3, 1, 3).unwrap();

        self.scene
            .light
            .clone_from_slice(&[[1., 1., 1.], [1., 1., 1.]]);
        self.scene.light_info.clone_from_slice(&[[1, 0, 0]]);

        layout_obj
            .add_buffer(0, true, self.scene.uniform.clone())
            .unwrap();
        layout_obj
            .add_buffer(0, false, self.scene.transforms.clone())
            .unwrap();
        layout_obj
            .add_buffer(1, false, self.scene.light.clone())
            .unwrap();
        layout_obj
            .add_buffer(2, false, self.scene.light_info.clone())
            .unwrap();

        let pipeline_render = layout_obj
            .clone()
            .create_graphics_pipeline(
                render_target.clone(),
                &shaders_obj,
                &VertexTexture::get_attributes(),
            )
            .unwrap();

        let layout_compute = graphics.create_layout(false, 0, 0, 1, 2).unwrap();

        self.scene.compute_buffer_in.clone_from_slice(
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
                .collect::<Vec<Particle>>(),
        );

        layout_compute
            .add_buffer(0, true, self.scene.uniform.clone())
            .unwrap();
        layout_compute
            .add_buffer(0, false, self.scene.compute_buffer_in.clone())
            .unwrap();
        layout_compute
            .add_buffer(1, false, self.scene.compute_buffer_out.clone())
            .unwrap();
        let pipeline_compute = layout_compute
            .clone()
            .create_compute_pipeline(&shader_compute)
            .unwrap();

        let mesh1 = Arc::new(
            Mesh::from_buffer(BufReader::new(
                File::open("resources/mishka/Untitled.obj").unwrap(),
            ))
            .unwrap(),
        );

        let obj1 = Object::with_mesh_textured(
            pipeline_render.clone(),
            mesh1.clone(),
            &[(0, test_sampler.clone())],
        );

        let obj2 = Object::with_mesh_textured(
            pipeline_render.clone(),
            mesh1.clone(),
            &[(0, default_sampler)],
        );

        let obj3 = Object::with_mesh_textured(
            pipeline_render.clone(),
            mesh1.clone(),
            &[(0, test_sampler)],
        );

        self.scene.objects.push(obj1);
        self.scene.objects.push(obj2);
        self.scene.objects.push(obj3);

        graphics.register_meshes(&self.scene.objects);
        layout_obj.register_samplers(&self.scene.objects).unwrap();

        self.graphics = Some(graphics);
        self.window = Some(window);
        self.layout_obj = Some(layout_obj);
        self.layout_compute = Some(layout_compute);
        self.pipeline_compute = Some(pipeline_compute);
    }

    fn about_to_wait(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        if let Some(now) = self.state.now {
            self.state.delta_time = now.elapsed();
            self.state.delta_time_sum += self.state.delta_time;
            self.state.current_frame += 1;

            if self.state.delta_time_sum.as_secs_f64() >= 1. {
                let pid = Pid::from_u32(std::process::id());

                self.system.refresh_process(pid);

                let process = self.system.process(pid).unwrap();

                println!("[DEBUG]");
                println!("FPS: {}", self.state.current_frame);
                println!("{}", self.graphics.as_ref().unwrap().update_debug_text());
                println!("RAM: {:.1} MB", process.memory() as f32 / 1024.);
                println!("CPU: {:.1}%", process.cpu_usage());
                println!();

                let compute = self.scene.compute_buffer_out.read_slice(0..5);
                for data in compute {
                    println!("{:?}", data);
                }
                println!();

                self.state.delta_time_sum = Duration::ZERO;
                self.state.current_frame = 0;
            }
        }

        self.state.now = Some(std::time::Instant::now());

        let transforms = [
            glam::Mat4::from_scale_rotation_translation(
                glam::Vec3::new(0.3, 0.3, 0.3),
                glam::Quat::from_mat4(&glam::Mat4::from_rotation_y(
                    PI * 2. * self.state.delta_time_sum.as_secs_f32(),
                )),
                glam::Vec3::new(0., 0., 0.),
            ),
            glam::Mat4::from_translation(glam::Vec3::new(0., 0., -3.)),
            glam::Mat4::from_translation(glam::Vec3::new(-3., 0., 1.)),
        ];

        let ubo = Uniform {
            eye: self.scene.camera.calc_eye_matrix(),
            time: self.state.startup.elapsed().as_secs_f32(),
        };

        self.scene.uniform.clone_from_slice(&[ubo.clone()]);
        self.scene.transforms.clone_from_slice(&transforms);

        self.layout_obj
            .clone()
            .unwrap()
            .flush_buffer_tasks()
            .unwrap();

        self.scene.uniform.clone_from_slice(&[ubo]);

        self.layout_compute
            .clone()
            .unwrap()
            .flush_buffer_tasks()
            .unwrap();

        self.graphics
            .clone()
            .unwrap()
            .dispatch(self.pipeline_compute.clone().unwrap(), [1, 1, 1])
            .unwrap();

        self.graphics
            .clone()
            .unwrap()
            .render_and_present(&self.scene.objects)
            .unwrap();

        // self.graphics
        //     .clone()
        //     .unwrap()
        //     .write_buffer_to_screen(vec![0u8; (100 * 100 * 4) as usize], (0, 0), (100, 100))
        //     .unwrap();
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
                // self.draw_text(self.graphics.clone().unwrap().update_debug_text());

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
    event_loop.set_control_flow(ControlFlow::Wait);

    let mut context = Context::new(settings)?;
    event_loop
        .run_app(&mut context)
        .expect("cannot run event loop");

    Ok(())
}
