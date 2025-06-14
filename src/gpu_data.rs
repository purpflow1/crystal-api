use std::alloc::{Layout, alloc};

use bytemuck::cast_slice;

#[derive(Clone)]
pub struct GpuVec(Vec<u8>);

impl GpuVec {
    pub fn new<T>(data: &[T]) -> Self {
        let size = data.len() * size_of::<T>();

        let layout = Layout::from_size_align(size, 0x10).expect("Failed to create layout");

        let ptr = unsafe { alloc(layout) as *mut u8 };
        if ptr.is_null() {
            panic!("Failed to allocate memory");
        }

        let mut tmp_vec = unsafe { Vec::<u8>::from_raw_parts(ptr, 0, size) };

        unsafe {
            tmp_vec.set_len(size);
            tmp_vec.copy_from_slice(std::slice::from_raw_parts(data.as_ptr() as *const u8, size));
        }

        Self(tmp_vec)
    }

    pub fn as_words(&self) -> &[u32] {
        cast_slice(&self.0)
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.0
    }
}

#[repr(C, align(16))]
pub struct AlignedValue<T>(T);

impl<T> AlignedValue<T> {
    pub const fn new(val: T) -> Self {
        Self(val)
    }
}
