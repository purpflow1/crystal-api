use std::{
    fmt::Debug,
    ops::{BitAnd, BitOr, BitXor},
};

macro_rules! wrap_bit_ops {
    ($bitstruct:tt) => {
        impl BitAnd for $bitstruct {
            type Output = Self;
            fn bitand(self, rhs: Self) -> Self::Output {
                Self(self.0 & rhs.0)
            }
        }

        impl BitOr for $bitstruct {
            type Output = Self;
            fn bitor(self, rhs: Self) -> Self::Output {
                Self(self.0 | rhs.0)
            }
        }

        impl BitXor for $bitstruct {
            type Output = Self;
            fn bitxor(self, rhs: Self) -> Self::Output {
                Self(self.0 ^ rhs.0)
            }
        }

        impl PartialEq for $bitstruct {
            fn eq(&self, other: &Self) -> bool {
                self.0 == other.0
            }
        }

        impl Eq for $bitstruct {}

        impl Debug for $bitstruct {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str(format!("0x{:08x}", self.0).as_str())
            }
        }

        impl $bitstruct {
            /// Returns true if all bits are zero
            pub fn is_none(&self) -> bool {
                self.0 == 0
            }
        }
    };
}

/// Buffer flags used to specify buffer usage
#[repr(transparent)]
#[derive(Default, Clone, Copy)]
pub struct BufferFlags(u8);
wrap_bit_ops!(BufferFlags);

impl BufferFlags {
    /// Uniform buffers are used to store small amount of data, non indexing
    pub const UNIFORM: Self = Self(0b001);
    /// Transfer buffers are used to read data from it.
    /// For example, after computations.
    pub const TRANSFER: Self = Self(0b010);
    /// Indicates that the buffer would be synced along the frame
    pub const SYNCED: Self = Self(0b100);
}
