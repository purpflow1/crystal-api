use crate::errors::GraphicsResult;
use crate::object::Object;
use crate::*;

const PARTICLES_GLSL: &str = "
#version 460

layout(set = 0, binding = 0) uniform UniformBufferObject {
    float time;
} ubo;

struct Particle {
    vec3 position;
    vec3 velocity;
};

layout(set = 1, binding = 0) readonly buffer ParticleSSBOIn {
    Particle data[];
} pin;

layout(set = 1, binding = 1) buffer ParticleSSBOOut {
    Particle data[];
} pout;

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

void main()
{
    uint index = gl_GlobalInvocationID.x;

    Particle p = pin.data[index];

    pout.data[index].position = p.position + p.velocity * ubo.time;
    pout.data[index].velocity = p.velocity;
}
";

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
#[derive(Debug, Clone, Copy)]
struct Particle {
    pos: Vector,
    vel: Vector,
}

const PARTICLE_NUM: u64 = 1024;

#[test]
fn compute_dispatching() -> GraphicsResult<()> {
    let device = Device::compute()?;
    let layout = device.create_layout(false, 0, 0, 1, 2)?;

    let mut buffer_uniform = device.create_buffer(1, true, false, false)?;
    let mut buffer_in = device.create_buffer::<Particle>(PARTICLE_NUM, false, true, false)?;
    let buffer_out = device.create_buffer::<Particle>(PARTICLE_NUM, false, true, false)?;

    layout.add_buffer(0, &buffer_uniform)?;
    layout.add_buffer(0, &buffer_in)?;
    layout.add_buffer(1, &buffer_out)?;

    let compiler = shaderc::Compiler::new().unwrap();
    let binary_result = compiler
        .compile_into_spirv(
            PARTICLES_GLSL,
            shaderc::ShaderKind::Compute,
            "particles.comp",
            "main",
            None,
        )
        .unwrap();

    let shader = Shader::from_bytes(binary_result.as_binary_u8(), ShaderStage::Compute)?;
    let pipeline = layout.create_compute_pipeline(&shader)?;

    let object = Object::compute(&pipeline, [PARTICLE_NUM as u32 / 256, 1, 1]);

    const TIME: f32 = 0.5;

    buffer_uniform[0] = Uniform { time: TIME };

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

    buffer_in[..].copy_from_slice(&input);

    device.dispatch_compute(&[object])?;

    for idx in 0..PARTICLE_NUM {
        let particle = buffer_out[idx];
        assert_eq!(particle.pos.x, idx as f32 + particle.vel.x * TIME);
        assert_eq!(particle.pos.y, particle.vel.y * TIME);
        assert_eq!(particle.pos.z, particle.vel.z * TIME);
    }

    Ok(())
}
