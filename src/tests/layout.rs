use crate::{errors::GraphicsResult, *};

#[test]
fn layout_non_empty_creation() -> GraphicsResult<()> {
    let device = Device::compute()?;
    device.create_layout(false, 0, 0, 0, 1)?;
    device.create_layout(false, 0, 0, 1, 0)?;
    device.create_layout(false, 0, 0, 1, 1)?;
    device.create_layout(false, 1, 1, 1, 1)?;
    device.create_layout(false, 2, 2, 2, 2)?;
    Ok(())
}

#[test]
fn layout_double_buffered_creation() -> GraphicsResult<()> {
    let device = Device::compute()?;
    device.create_layout(true, 0, 0, 0, 1)?;
    device.create_layout(true, 0, 0, 1, 0)?;
    device.create_layout(true, 0, 0, 1, 1)?;
    device.create_layout(true, 1, 1, 1, 1)?;
    device.create_layout(true, 2, 2, 2, 2)?;
    Ok(())
}

#[test]
#[should_panic(expected = "layout")]
fn layout_empty_buffers_creation() {
    let device = Device::compute().expect("cannot create device");
    device.create_layout(false, 0, 0, 0, 0).expect("layout");
}

#[test]
fn layout_texture_no_samplers_creation() {
    let device = Device::compute().expect("cannot create device");
    if device.create_layout(false, 1, 0, 1, 1).is_ok() {
        panic!("invalid")
    }
}

#[test]
fn layout_sampler_no_textures_creation() {
    let device = Device::compute().expect("cannot create device");
    if device.create_layout(false, 0, 1, 1, 1).is_ok() {
        panic!("invalid")
    }
}
