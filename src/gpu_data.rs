use std::{
    mem::{ManuallyDrop, MaybeUninit},
    ops::Range,
    sync::{Arc, Mutex, RwLock},
};

use crate::traits;

pub trait IntoGpuTexture {
    fn get_texture(&self) -> Arc<dyn traits::Texture>;
    fn get_alive(&self) -> Arc<RwLock<bool>>;
}

pub struct PtrHandler(pub(crate) RwLock<*mut u8>);

unsafe impl Sync for PtrHandler {}
unsafe impl Send for PtrHandler {}

impl PtrHandler {
    pub fn new_null() -> Self {
        Self(RwLock::new(std::ptr::null_mut()))
    }

    pub fn get_ptr<T>(&self) -> *mut T {
        *self.0.write().unwrap() as *mut T
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

#[derive(Clone, Debug)]
pub struct GpuBufferTask {
    src: Vec<u8>,
}

unsafe impl Sync for GpuBufferTask {}
unsafe impl Send for GpuBufferTask {}

impl GpuBufferTask {
    pub(crate) async fn flush<'a>(&self, dst: *mut u8) {
        let dst = unsafe { std::slice::from_raw_parts_mut(dst, self.src.len()) };
        dst.copy_from_slice(&self.src);
    }
}

pub trait IntoGpuBuffer: Sync + Send {
    fn size(&self) -> u64;
    fn get_ptr(&self) -> Arc<PtrHandler>;
    fn query_tasks(&self) -> Vec<GpuBufferTask>;
    fn is_transfer(&self) -> bool;
}

pub struct GpuVec<T> {
    ptr: Arc<PtrHandler>,
    len: usize,
    tasks: Mutex<Vec<GpuBufferTask>>,
    transfer: bool,
    _typ: MaybeUninit<T>,
}

impl<T> Drop for GpuVec<T> {
    fn drop(&mut self) {
        *self.ptr.0.write().unwrap() = std::ptr::null_mut();
    }
}

impl<T: Sync + Send> IntoGpuBuffer for GpuVec<T> {
    fn size(&self) -> u64 {
        (self.len * size_of::<T>()) as u64
    }

    fn get_ptr(&self) -> Arc<PtrHandler> {
        self.ptr.clone()
    }

    fn query_tasks(&self) -> Vec<GpuBufferTask> {
        let mut tasks = self.tasks.lock().unwrap();
        let tasks_cloned = (*tasks).clone();
        tasks.clear();
        tasks_cloned
    }

    fn is_transfer(&self) -> bool {
        self.transfer
    }
}

impl<T> GpuVec<T> {
    pub fn with_len(len: usize) -> Arc<Self> {
        Arc::new(Self::new_in(len))
    }

    pub fn with_len_transfer(len: usize) -> Arc<Self> {
        let mut gpu_vec = Self::new_in(len);
        gpu_vec.transfer = true;
        Arc::new(gpu_vec)
    }

    fn new_in(len: usize) -> Self {
        Self {
            ptr: Arc::new(PtrHandler::new_null()),
            len,
            tasks: Mutex::new(Vec::new()),
            transfer: false,
            _typ: MaybeUninit::uninit(),
        }
    }

    pub fn read(&self) -> &[T] {
        self.read_slice(0..self.len)
    }

    pub fn read_slice(&self, range: Range<usize>) -> &[T] {
        assert!(self.transfer, "fatal: trying to read non-transfer GpuVec");
        assert!(range.end <= self.len);

        let lock = self.ptr.0.read().unwrap();
        let addr = *lock as usize;
        let start = addr + range.start;

        unsafe { std::slice::from_raw_parts(start as *const T, range.len()) }
    }

    pub fn clone_from_slice(&self, data: &[T])
    where
        T: Clone,
    {
        self.clone_from_slice_with_offset(0, data);
    }

    pub fn clone_from_slice_with_offset(&self, offset: usize, data: &[T])
    where
        T: Clone,
    {
        let size = data.len() * size_of::<T>();
        let manually_drop = ManuallyDrop::new(data.to_vec());
        let tmp = unsafe {
            Vec::from_raw_parts(manually_drop.as_ptr().add(offset) as *mut u8, size, size)
        };
        self.tasks.lock().unwrap().push(GpuBufferTask { src: tmp });
    }
}
