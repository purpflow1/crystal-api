#[derive(Clone)]
#[repr(transparent)]
pub struct GpuVec(Vec<u8>);

impl GpuVec {
    pub fn new<T>(data: &[T]) -> Self {
        let size = data.len() * size_of::<T>();
        let mut tmp_v = Vec::<u8>::with_capacity(size);
        unsafe {
            tmp_v.set_len(size);
            (data.as_ptr() as *const u8).copy_to(tmp_v.as_ptr() as *mut u8, size);
        };

        Self(tmp_v)
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
