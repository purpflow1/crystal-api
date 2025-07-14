use std::sync::{Arc, Mutex, RwLock};

use crate::traits;

pub trait AsBytes {
    fn as_bytes(&self) -> &[u8];
}

impl<T> AsBytes for Vec<T> {
    fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(self.as_ptr() as *const u8, self.len() * size_of::<T>())
        }
    }
}

impl<T> AsBytes for &[T] {
    fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(self.as_ptr() as *const u8, self.len() * size_of::<T>())
        }
    }
}

pub trait IntoGpuTexture {
    fn get_texture(&self) -> Arc<dyn traits::Texture>;
    fn get_alive(&self) -> Arc<RwLock<bool>>;
}

pub struct PtrHandler(pub(crate) Mutex<*mut u8>);

unsafe impl Sync for PtrHandler {}
unsafe impl Send for PtrHandler {}

impl PtrHandler {
    pub fn new_null() -> Self {
        Self(Mutex::new(std::ptr::null_mut()))
    }

    pub fn get_ptr<T>(&self) -> *mut T {
        *self.0.lock().unwrap() as *mut T
    }
}

pub struct GpuSampler {
    pub texture: Arc<dyn traits::Texture>,
    alive: Arc<RwLock<bool>>,
}

impl Drop for GpuSampler {
    fn drop(&mut self) {
        *self.alive.write().unwrap() = false;
    }
}

impl IntoGpuTexture for GpuSampler {
    fn get_alive(&self) -> Arc<RwLock<bool>> {
        self.alive.clone()
    }

    fn get_texture(&self) -> Arc<dyn traits::Texture> {
        self.texture.clone()
    }
}

impl GpuSampler {
    pub fn from_texture(texture: Arc<dyn traits::Texture>) -> Arc<Self> {
        Arc::new(Self {
            texture,
            alive: Arc::new(RwLock::new(true)),
        })
    }
}
