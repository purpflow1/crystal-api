use std::{
    fs::File,
    io::{BufReader, Read},
};

use crystal_api::{
    AsBytes, Shader, ShaderStage,
    debug::{LoggingLevel, set_internal_logging_level},
    errors::GraphicsResult,
    init_api_instance,
    object::Object,
};

#[repr(C, align(16))]
struct Uniform {
    time: f32,
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug)]
struct Vector {
    x: f32,
    y: f32,
    z: f32,
}

#[repr(C, align(16))]
#[derive(Debug)]
struct Particle {
    pos: Vector,
    vel: Vector,
}

const PARTICLE_NUM: u64 = 1024 * 1024;

fn main() -> GraphicsResult<()> {
    set_internal_logging_level(LoggingLevel::Console);

    let api = init_api_instance()?;
    let layout = api.create_layout(false, 0, 0, 1, 2)?;

    let buffer_uniform = api.create_buffer(size_of::<Uniform>() as u64, true, false, false)?;
    let buffer_in = api.create_buffer(
        size_of::<Particle>() as u64 * PARTICLE_NUM,
        false,
        true,
        false,
    )?;
    let buffer_out = api.create_buffer(
        size_of::<Particle>() as u64 * PARTICLE_NUM,
        false,
        true,
        false,
    )?;

    layout.add_buffer(0, buffer_uniform.clone())?;
    layout.add_buffer(0, buffer_in.clone())?;
    layout.add_buffer(1, buffer_out.clone())?;

    println!("compiling GLSL shader...");
    let file_name = "examples/compute-test/shaders/particles.comp";
    let mut source = String::new();
    let mut reader = BufReader::new(File::open(file_name).unwrap());
    reader.read_to_string(&mut source).unwrap();

    let compiler = shaderc::Compiler::new().unwrap();
    let mut options = shaderc::CompileOptions::new().unwrap();
    options.add_macro_definition("EP", Some("main"));
    let binary_result = compiler
        .compile_into_spirv(
            source.as_str(),
            shaderc::ShaderKind::Compute,
            file_name,
            "main",
            Some(&options),
        )
        .unwrap();

    let shader = Shader::from_bytes(binary_result.as_binary_u8(), ShaderStage::Compute)?;
    let pipeline = layout.create_compute_pipeline(&shader)?;

    let object = Object::compute(pipeline, [PARTICLE_NUM as u32 / 256, 1, 1]);

    buffer_uniform
        .get_memory_full()
        .copy_from_slice(vec![Uniform { time: 0.5 }].as_bytes());

    let vel = Vector {
        x: 0.1,
        y: -0.5,
        z: 2.3,
    };
    let input: Vec<_> = (0..PARTICLE_NUM)
        .map(|x| Particle {
            pos: Vector {
                x: x as f32,
                y: 0.,
                z: 0.,
            },
            vel,
        })
        .collect();

    buffer_in
        .get_memory_full()
        .copy_from_slice(input.as_bytes());

    let now = std::time::Instant::now();
    api.dispatch_compute(&[object])?;
    println!("dispatch ended in {:.2} secs", now.elapsed().as_secs_f32());

    let out = buffer_out.get_memory_full();

    for offset in (0..4 * size_of::<Particle>()).step_by(size_of::<Particle>()) {
        let ptr = (&out[offset]) as *const u8 as *const Particle;
        let particle = unsafe { ptr.read() };
        println!(
            "[ {} {} {} ] [ {} {} {} ]",
            particle.pos.x,
            particle.pos.y,
            particle.pos.z,
            particle.vel.x,
            particle.vel.y,
            particle.vel.z
        )
    }

    Ok(())
}
