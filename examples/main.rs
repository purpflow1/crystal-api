#![feature(random)]

use crystal_api::{errors::CrystalResult, object::Object, vulkan::VulkanEntry, *};

use std::{
    cell::RefCell,
    f32::consts::PI,
    fs::File,
    io::BufReader,
    iter::zip,
    path::Path,
    random::random,
    sync::Arc,
    time::{Duration, SystemTime},
};

use images::Image2D;
use mesh::{Mesh, VertexTexture};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::WindowEvent,
    event_loop::{ControlFlow, EventLoop},
    platform::x11::WindowAttributesExtX11,
    window::Window,
};

const MAX_INSTANCE_NUM: u64 = 128;
const IMAGE_SAMPLED_NUM: u64 = 8;

struct State {
    delta_time: Duration,
    delta_time_sum: Duration,
    current_frame: u16,
    startup_time: SystemTime,
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
    ambient_lights: Vec<glam::Vec4>,
    point_lights: Vec<(glam::Vec4, glam::Vec4)>,
    direct_lights: Vec<(glam::Vec4, glam::Vec4)>,
    objects_pbr: Vec<Arc<RefCell<Object>>>,
}

struct Context {
    window: Option<Window>,
    graphics: Option<Box<dyn GraphicsApi>>,

    layout_pbr: Option<Arc<RefCell<dyn Layout>>>,

    settings: GraphicsApiInitSettings,
    scene: Scene,

    state: State,
}

impl Context {
    pub fn new(settings: GraphicsApiInitSettings) -> CrystalResult<Self> {
        let width = settings.width;
        let height = settings.height;

        Ok(Self {
            window: None,
            graphics: None,

            layout_pbr: None,

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
                ambient_lights: vec![glam::Vec4::new(0.1, 0.1, 0.1, 0.)],
                point_lights: vec![(
                    glam::Vec4::new(1., 1., 1., 0.),
                    glam::Vec4::new(0., 5., 0., 0.),
                )],
                direct_lights: vec![],
                objects_pbr: vec![],
            },

            state: State {
                delta_time: Duration::ZERO,
                delta_time_sum: Duration::ZERO,
                current_frame: 0,
                startup_time: SystemTime::now(),
            },
        })
    }
}

impl ApplicationHandler for Context {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let window = event_loop
            .create_window(
                Window::default_attributes()
                    .with_base_size(LogicalSize::new(self.settings.width, self.settings.height)),
            )
            .unwrap();

        let graphics = VulkanEntry::with_presentation(
            &self.settings,
            (
                window.display_handle().unwrap().as_raw(),
                window.window_handle().unwrap().as_raw(),
            ),
        )
        .expect("cannot create vulkan entry");

        let shaders_pbr = [
            Shader::open("shaders/desc.vert.spv", ShaderStage::Vertex).unwrap(),
            Shader::open("shaders/desc.frag.spv", ShaderStage::Fragment).unwrap(),
        ];

        let render_target = graphics.get_viewport();

        let default_texture_image =
            Image2D::new(Path::new("resources/textures/default.png")).unwrap();

        let default_texture_map = graphics
            .create_texture(&default_texture_image, 1.0)
            .unwrap();

        let test_texture_image = Image2D::new(Path::new("resources/textures/test.png")).unwrap();

        let test_texture_map = graphics.create_texture(&test_texture_image, 1.0).unwrap();

        let layout_pbr = graphics
            .create_layout(
                self.settings.viewport_frames_in_flight,
                IMAGE_SAMPLED_NUM as u32,
                MAX_INSTANCE_NUM,
                &[
                    (
                        true,
                        size_of::<glam::Mat4>() as u64 + size_of::<f32>() as u64 + 12, // because of 16 bit alignment
                    ),
                    (false, size_of::<glam::Mat4>() as u64 * MAX_INSTANCE_NUM), // model data
                    (false, size_of::<glam::Vec4>() as u64 * 60),               // light
                    (false, size_of::<u32>() as u64 * 3),                       // light data
                    (
                        false,
                        MAX_INSTANCE_NUM * size_of::<u32>() as u64 * IMAGE_SAMPLED_NUM,
                    ), // texture data
                ],
            )
            .unwrap();

        let pipeline_pbr = render_target
            .borrow()
            .create_graphics_pipeline(
                layout_pbr.borrow(),
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
            &[(0, test_texture_map.clone())],
        );

        let obj2 = Object::with_mesh_textured(
            pipeline_pbr.clone(),
            mesh1.clone(),
            &[(0, default_texture_map.clone())],
        );

        let obj3 = Object::with_mesh_textured(
            pipeline_pbr.clone(),
            mesh1.clone(),
            &[(0, test_texture_map)],
        );

        self.scene.objects_pbr.push(obj1);
        self.scene.objects_pbr.push(obj2);
        self.scene.objects_pbr.push(obj3);

        self.graphics = Some(graphics);
        self.window = Some(window);
        self.layout_pbr = Some(layout_pbr);
    }

    fn about_to_wait(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let now = std::time::Instant::now();

        let graphics = self.graphics.as_ref().unwrap();

        let viewport = graphics.get_viewport();
        let current_frame = viewport.borrow().get_current_frame();

        let mut layout_pbr = self.layout_pbr.as_ref().unwrap().borrow_mut();

        let mut transforms = vec![];

        let mut iter: Vec<Arc<RefCell<Object>>> = vec![];

        // let left: usize = random::<usize>() % self.scene.objects_pbr.len();
        // let right = random::<usize>() % (self.scene.objects_pbr.len() - left) + left + 1;

        for idx in 0..3 {
            iter.push(self.scene.objects_pbr[idx].clone());
        }

        if random::<usize>() % 2 == 0 {
            iter = iter.iter().rev().map(|x| x.clone()).collect();
        }

        for (idx, object) in zip(0..self.scene.objects_pbr.len(), iter) {
            let obj = object.borrow();

            layout_pbr.add_object_to_queue(object.clone());

            let transform = match idx {
                0 => glam::Mat4::from_scale_rotation_translation(
                    glam::Vec3::new(0.3, 0.3, 0.3),
                    glam::Quat::from_mat4(&glam::Mat4::from_rotation_y(
                        PI * 2. * self.state.delta_time_sum.as_secs_f32(),
                    )),
                    glam::Vec3::new(0., 0., 0.),
                ),
                1 => glam::Mat4::from_translation(glam::Vec3::new(0., 0., -3.)),
                2 => glam::Mat4::from_translation(glam::Vec3::new(-3., 0., 1.)),
                _ => glam::Mat4::IDENTITY,
            };

            transforms.push(transform);

            match &obj.textures {
                None => {}
                Some(textures) => {
                    for image_idx in 0..IMAGE_SAMPLED_NUM as u32 {
                        layout_pbr
                            .write_to_buffer(
                                false,
                                current_frame,
                                3,
                                (idx as u32 * IMAGE_SAMPLED_NUM as u32 + image_idx) as usize,
                                GpuVec::new(&[
                                    if textures
                                        .iter()
                                        .find(|texture| texture.0 == image_idx)
                                        .is_some()
                                    {
                                        1u32
                                    } else {
                                        0u32
                                    },
                                ]),
                            )
                            .unwrap();
                    }
                }
            };
        }

        let ubo = (
            self.scene.camera.calc_eye_matrix(),
            self.state.startup_time.elapsed().unwrap().as_secs_f32(),
        );

        layout_pbr
            .write_to_buffer(true, current_frame, 0, 0, GpuVec::new(&[ubo]))
            .expect("cannot update UBO");

        layout_pbr
            .write_to_buffer(false, current_frame, 0, 0, GpuVec::new(&transforms))
            .expect("cannot update SSBO");

        layout_pbr
            .write_to_buffer(
                false,
                current_frame,
                1,
                0,
                GpuVec::new(&self.scene.ambient_lights),
            )
            .expect("cannot update LBO");

        layout_pbr
            .write_to_buffer(
                false,
                current_frame,
                1,
                self.scene.ambient_lights.len(),
                GpuVec::new(&self.scene.direct_lights),
            )
            .expect("cannot update LBO");

        layout_pbr
            .write_to_buffer(
                false,
                current_frame,
                1,
                self.scene.ambient_lights.len() + self.scene.direct_lights.len(),
                GpuVec::new(&self.scene.point_lights),
            )
            .expect("cannot update LBO");

        let info_light = [
            self.scene.ambient_lights.len() as u32,
            self.scene.direct_lights.len() as u32,
            self.scene.point_lights.len() as u32,
        ];

        layout_pbr
            .write_to_buffer(false, current_frame, 2, 0, GpuVec::new(&info_light))
            .expect("cannot update ILBO");

        drop(layout_pbr);

        graphics
            .render(&[self.layout_pbr.as_ref().unwrap().clone()])
            .unwrap();

        if self.settings.max_fps != 0 && !event_loop.exiting() {
            let time_to_sleep = Duration::from_secs_f64(1. / self.settings.max_fps as f64)
                .checked_sub(now.elapsed())
                .unwrap_or(Duration::ZERO);

            std::thread::sleep(time_to_sleep);
        }

        self.state.delta_time = now.elapsed();
        self.state.delta_time_sum += self.state.delta_time;
        self.state.current_frame += 1;

        if self.state.delta_time_sum.as_secs_f64() >= 1. {
            // println!(
            //     "FPS: {}",
            //     (1. / (self.state.delta_time_sum.as_secs_f64() / self.settings.max_fps as f64))
            //         as u16
            // );
            self.state.delta_time_sum = Duration::ZERO;
        }
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
            WindowEvent::RedrawRequested => {
                self.window.as_ref().unwrap().request_redraw();
            }
            _ => {}
        }
    }
}

fn main() -> CrystalResult<()> {
    let settings = GraphicsApiInitSettings::default()
        .viewport_frames_in_flight(2)
        .msaa_samples(8)
        .max_fps(60)
        .width(1000)
        .height(700);

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.set_control_flow(ControlFlow::Wait);

    let mut context = Context::new(settings)?;
    event_loop
        .run_app(&mut context)
        .expect("cannot run event loop");

    Ok(())
}
