use crate::{errors::GraphicsResult, *};

#[test]
fn buffer_uniform_creation() -> GraphicsResult<()> {
    let device = Device::compute()?;
    device.create_buffer::<u8>(1024 * 1024, true, false, false)?;
    Ok(())
}

#[test]
fn buffer_uniform_sync_creation() -> GraphicsResult<()> {
    let device = Device::compute()?;
    device.create_buffer::<u8>(1024 * 1024, true, false, true)?;
    Ok(())
}

#[test]
fn buffer_storage_creation() -> GraphicsResult<()> {
    let device = Device::compute()?;
    device.create_buffer::<u8>(1024 * 1024, false, false, false)?;
    Ok(())
}

#[test]
fn buffer_storage_sync_creation() -> GraphicsResult<()> {
    let device = Device::compute()?;
    device.create_buffer::<u8>(1024 * 1024, false, false, true)?;
    Ok(())
}

#[test]
fn buffer_transfer_creation() -> GraphicsResult<()> {
    let device = Device::compute()?;
    device.create_buffer::<u8>(1024 * 1024, false, true, false)?;
    Ok(())
}

#[test]
fn buffer_transfer_sync_creation() -> GraphicsResult<()> {
    let device = Device::compute()?;
    device.create_buffer::<u8>(1024 * 1024, false, true, true)?;
    Ok(())
}

#[test]
fn buffer_binding() -> GraphicsResult<()> {
    let device = Device::compute()?;
    let buffer_storage = device.create_buffer::<u8>(1024, false, false, false)?;
    let buffer_uniform = device.create_buffer::<u8>(1024, true, false, false)?;
    let layout = device.create_layout(false, 0, 0, 1, 2)?;
    layout.add_buffer(0, &buffer_storage)?;
    layout.add_buffer(1, &buffer_storage)?;
    layout.add_buffer(0, &buffer_uniform)?;
    Ok(())
}

#[test]
fn buffer_rebinding() -> GraphicsResult<()> {
    let device = Device::compute()?;
    let buffer_storage = device.create_buffer::<u8>(1024, false, false, false)?;
    let layout = device.create_layout(false, 0, 0, 0, 1)?;
    layout.add_buffer(0, &buffer_storage)?;
    layout.add_buffer(0, &buffer_storage)?;
    Ok(())
}

// TODO handle
// #[test]
// #[should_panic]
// fn buffer_binding_with_overflow() {
//     let device = init_api_device().unwrap();
//     let buffer_storage = device.create_buffer(1024, false, false, false).unwrap();
//     let layout = device.create_layout(false, 0, 0, 0, 1).unwrap();
//     layout.add_buffer(0, buffer_storage.clone()).unwrap();
//     layout.add_buffer(1, buffer_storage).unwrap();
// }
