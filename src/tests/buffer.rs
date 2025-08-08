use crate::*;

#[test]
fn buffer_uniform_creation() -> GraphicsResult<()> {
    let instance = init_api_instance()?;
    instance.create_buffer(1024 * 1024, true, false, false)?;
    Ok(())
}

#[test]
fn buffer_uniform_sync_creation() -> GraphicsResult<()> {
    let instance = init_api_instance()?;
    instance.create_buffer(1024 * 1024, true, false, true)?;
    Ok(())
}

#[test]
fn buffer_storage_creation() -> GraphicsResult<()> {
    let instance = init_api_instance()?;
    instance.create_buffer(1024 * 1024, false, false, false)?;
    Ok(())
}

#[test]
fn buffer_storage_sync_creation() -> GraphicsResult<()> {
    let instance = init_api_instance()?;
    instance.create_buffer(1024 * 1024, false, false, true)?;
    Ok(())
}

#[test]
fn buffer_transfer_creation() -> GraphicsResult<()> {
    let instance = init_api_instance()?;
    instance.create_buffer(1024 * 1024, false, true, false)?;
    Ok(())
}

#[test]
fn buffer_transfer_sync_creation() -> GraphicsResult<()> {
    let instance = init_api_instance()?;
    instance.create_buffer(1024 * 1024, false, true, true)?;
    Ok(())
}

#[test]
fn buffer_binding() -> GraphicsResult<()> {
    let instance = init_api_instance()?;
    let buffer_storage = instance.create_buffer(1024, false, false, false)?;
    let buffer_uniform = instance.create_buffer(1024, true, false, false)?;
    let layout = instance.create_layout(false, 0, 0, 1, 2)?;
    layout.add_buffer(0, buffer_storage.clone())?;
    layout.add_buffer(1, buffer_storage)?;
    layout.add_buffer(0, buffer_uniform)?;
    Ok(())
}

#[test]
fn buffer_rebinding() -> GraphicsResult<()> {
    let instance = init_api_instance()?;
    let buffer_storage = instance.create_buffer(1024, false, false, false)?;
    let layout = instance.create_layout(false, 0, 0, 0, 1)?;
    layout.add_buffer(0, buffer_storage.clone())?;
    layout.add_buffer(0, buffer_storage)?;
    Ok(())
}

// TODO handle
// #[test]
// #[should_panic]
// fn buffer_binding_with_overflow() {
//     let instance = init_api_instance().unwrap();
//     let buffer_storage = instance.create_buffer(1024, false, false, false).unwrap();
//     let layout = instance.create_layout(false, 0, 0, 0, 1).unwrap();
//     layout.add_buffer(0, buffer_storage.clone()).unwrap();
//     layout.add_buffer(1, buffer_storage).unwrap();
// }
