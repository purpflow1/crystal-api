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

struct Scene {
    camera: Camera,

    light: Arc<GpuVec<[f32; 3]>>,
    light_info: Arc<GpuVec<[u32; 3]>>,

    uniform: Arc<GpuVec<(glam::Mat4, f32)>>,
    transforms: Arc<GpuVec<glam::Mat4>>,

    objects_pbr: Vec<Arc<Object>>,
}

struct Context {
    window: Option<Window>,
    graphics: Option<Arc<VulkanEntry>>,

    layout_pbr: Option<Arc<dyn Layout>>,

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
            layout_pbr: None,
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

                uniform: GpuVec::with_len(1),
                objects_pbr: vec![],
            },

            state: State {
                delta_time: Duration::ZERO,
                delta_time_sum: Duration::ZERO,
                current_frame: 0,
                now: None,
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
            .expect("cannot create vulkan entry"); // TODO not panic

        let shaders_pbr = [
            Shader::open("shaders/desc.vert.spv", ShaderStage::Vertex).unwrap(),
            Shader::open("shaders/desc.frag.spv", ShaderStage::Fragment).unwrap(),
        ];

        let render_target = graphics.get_viewport();

        let default_texture_image =
            Image2D::new(Path::new("resources/textures/default.png")).unwrap();
        let default_sampler = graphics
            .create_sampler(&default_texture_image, 1.0)
            .unwrap();

        let test_texture_image = Image2D::new(Path::new("resources/textures/test.png")).unwrap();
        let test_sampler = graphics.create_sampler(&test_texture_image, 1.0).unwrap();

        let layout_pbr = graphics.create_layout(2, 3, 1, 3).unwrap();

        self.scene
            .light
            .clone_from_slice(&[[1., 1., 1.], [1., 1., 1.]]);
        self.scene.light_info.clone_from_slice(&[[1, 0, 0]]);

        layout_pbr
            .add_buffer(0, true, self.scene.uniform.clone())
            .unwrap();
        layout_pbr
            .add_buffer(0, false, self.scene.transforms.clone())
            .unwrap();
        layout_pbr
            .add_buffer(1, false, self.scene.light.clone())
            .unwrap();
        layout_pbr
            .add_buffer(2, false, self.scene.light_info.clone())
            .unwrap();

        let pipeline_pbr = render_target
            .create_graphics_pipeline(
                layout_pbr.clone(),
                &shaders_pbr,
                &VertexTexture::get_attributes(),
            )
            .unwrap();

        let mesh1 = Arc::new(
            Mesh::from_buffer(BufReader::new(
                File::open("resources/mishka/Untitled.obj").unwrap(),
            ))
            .unwrap(),
        );

        let obj1 = Object::with_mesh_textured(
            pipeline_pbr.clone(),
            mesh1.clone(),
            &[(0, test_sampler.clone())],
        );

        let obj2 = Object::with_mesh_textured(
            pipeline_pbr.clone(),
            mesh1.clone(),
            &[(0, default_sampler)],
        );

        let obj3 =
            Object::with_mesh_textured(pipeline_pbr.clone(), mesh1.clone(), &[(0, test_sampler)]);

        self.scene.objects_pbr.push(obj1);
        self.scene.objects_pbr.push(obj2);
        self.scene.objects_pbr.push(obj3);

        graphics.register_meshes(&self.scene.objects_pbr);
        layout_pbr
            .register_samplers(&self.scene.objects_pbr)
            .unwrap();

        self.graphics = Some(graphics);
        self.window = Some(window);
        self.layout_pbr = Some(layout_pbr);
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

        let ubo = (
            self.scene.camera.calc_eye_matrix(),
            self.state.now.unwrap().elapsed().as_secs_f32(),
        );

        self.scene.uniform.clone_from_slice(&[ubo]);
        self.scene.transforms.clone_from_slice(&transforms);

        self.graphics
            .clone()
            .unwrap()
            .render_and_present(&self.scene.objects_pbr)
            .unwrap();

        self.graphics
            .clone()
            .unwrap()
            .write_buffer_to_screen(vec![0u8; (100 * 100 * 4) as usize], (0, 0), (100, 100))
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
        .double_buffering(true)
        .vsync(false)
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
