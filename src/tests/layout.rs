use crate::*;

#[test]
fn layout_non_empty_creation() -> GraphicsResult<()> {
    let instance = init_api_instance()?;
    instance.create_layout(false, 0, 0, 0, 1)?;
    instance.create_layout(false, 0, 0, 1, 0)?;
    instance.create_layout(false, 0, 0, 1, 1)?;
    instance.create_layout(false, 1, 1, 1, 1)?;
    instance.create_layout(false, 2, 2, 2, 2)?;
    Ok(())
}

#[test]
fn layout_double_buffered_creation() -> GraphicsResult<()> {
    let instance = init_api_instance()?;
    instance.create_layout(true, 0, 0, 0, 1)?;
    instance.create_layout(true, 0, 0, 1, 0)?;
    instance.create_layout(true, 0, 0, 1, 1)?;
    instance.create_layout(true, 1, 1, 1, 1)?;
    instance.create_layout(true, 2, 2, 2, 2)?;
    Ok(())
}

#[test]
#[should_panic]
fn layout_empty_buffers_creation() {
    let instance = init_api_instance().expect("cannot create instance");
    instance.create_layout(false, 0, 0, 0, 0).unwrap();
}

#[test]
#[should_panic]
fn layout_texture_no_samplers_creation() {
    let instance = init_api_instance().expect("cannot create instance");
    instance.create_layout(false, 1, 0, 1, 1).unwrap();
}

#[test]
#[should_panic]
fn layout_sampler_no_textures_creation() {
    let instance = init_api_instance().expect("cannot create instance");
    instance.create_layout(false, 0, 1, 1, 1).unwrap();
}
