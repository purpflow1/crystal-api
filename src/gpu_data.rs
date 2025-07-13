use std::sync::{Arc, Mutex, RwLock};

use crate::{Buffer, traits};

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

pub struct EmptyBuffer;
pub struct EmptyGpuVec;

impl traits::GpuVec for EmptyGpuVec {
    fn read(&self) -> &[u8] {
        &[]
    }

    fn copy_from_slice(&mut self, _data: &[u8]) {}
}

impl traits::Buffer for EmptyBuffer {
    fn get_memory(&self) -> Arc<Mutex<dyn crate::GpuVec>> {
        Arc::new(Mutex::new(EmptyGpuVec))
    }
}

impl EmptyBuffer {
    pub fn new() -> Arc<dyn Buffer> {
        Arc::new(Self)
    }
}

// pub trait IntoGpuBuffer: Sync + Send {
//     fn size(&self) -> u64;
//     fn get_ptrs(&self) -> Vec<Arc<PtrHandler>>;
//     fn is_transfer(&self) -> bool;
//     fn is_uniform(&self) -> bool;
// }

// pub struct GpuVec<T> {
//     ptrs: Vec<Arc<PtrHandler>>,
//     n_pass: Mutex<usize>,
//     len: usize,
//     uniform: bool,
//     transfer: bool,
//     _typ: MaybeUninit<T>,
// }

// impl<T> Drop for GpuVec<T> {
//     fn drop(&mut self) {
//         self.ptrs
//             .iter()
//             .for_each(|ptr| *ptr.0.lock().unwrap() = std::ptr::null_mut());
//     }
// }

// impl<T: Sync + Send> IntoGpuBuffer for GpuVec<T> {
//     fn size(&self) -> u64 {
//         (self.len * size_of::<T>()) as u64
//     }

//     fn get_ptrs(&self) -> Vec<Arc<PtrHandler>> {
//         self.ptrs.clone()
//     }

//     fn is_transfer(&self) -> bool {
//         self.transfer
//     }

//     fn is_uniform(&self) -> bool {
//         self.uniform
//     }
// }

// impl<T> GpuVec<T> {
//     pub fn storage(len: usize) -> Arc<Self> {
//         Arc::new(Self::new_in(len))
//     }

//     pub fn uniform(len: usize) -> Arc<Self> {
//         let mut gpu_vec = Self::new_in(len);
//         gpu_vec.uniform = true;
//         Arc::new(gpu_vec)
//     }

//     pub fn transfer(len: usize) -> Arc<Self> {
//         let mut gpu_vec = Self::new_in(len);
//         gpu_vec.transfer = true;
//         Arc::new(gpu_vec)
//     }

//     fn new_in(len: usize) -> Self {
//         Self {
//             ptrs: vec![Arc::new(PtrHandler::new_null())],
//             len,
//             transfer: false,
//             uniform: false,
//             n_pass: Mutex::new(0),
//             _typ: MaybeUninit::uninit(),
//         }
//     }

//     pub fn read(&self) -> &[T] {
//         self.read_slice(0..self.len)
//     }

//     pub fn read_slice(&self, range: Range<usize>) -> &[T] {
//         assert!(self.transfer, "fatal: trying to read non-transfer GpuVec");
//         assert!(range.end <= self.len);

//         let mut n_pass = self.n_pass.lock().unwrap();

//         let lock = self.ptrs[*n_pass].0.lock().unwrap();
//         let addr = *lock as usize;
//         let start = addr + range.start;

//         *n_pass = (*n_pass + 1) % self.ptrs.len();

//         unsafe { std::slice::from_raw_parts(start as *const T, range.len()) }
//     }

//     pub fn clone_from_slice(&self, data: &[T])
//     where
//         T: Clone,
//     {
//         self.clone_from_slice_with_offset(data, 0);
//     }

//     pub fn clone_from_slice_with_offset(&self, data: &[T], offset: usize)
//     where
//         T: Clone,
//     {
//         let mut n_pass = self.n_pass.lock().unwrap();
//         let size = data.len() * size_of::<T>();
//         let manually_drop = ManuallyDrop::new(data.to_vec());
//         let src = unsafe {
//             Vec::from_raw_parts(manually_drop.as_ptr().add(offset) as *mut u8, size, size)
//         };
//         let dst = unsafe { std::slice::from_raw_parts_mut(self.ptrs[*n_pass].get_ptr(), size) };
//         dst.copy_from_slice(&src);
//         *n_pass = (*n_pass + 1) % self.ptrs.len();
//     }
// }
