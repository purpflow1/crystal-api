/// Represents internal wrapper logging level
#[derive(Clone, Copy)]
pub enum LoggingLevel {
    /// Console output only
    Console,
    /// No logging
    None,
}

pub(crate) static LOGGING_LEVEL: RwLock<LoggingLevel> = RwLock::new(LoggingLevel::None);
static STARTUP_TIME: RwLock<MaybeUninit<Instant>> = RwLock::new(MaybeUninit::uninit());

/// Sets logging level
pub fn set_internal_logging_level(logging_level: LoggingLevel) {
    *LOGGING_LEVEL.write().unwrap() = logging_level
}

pub(crate) fn get_logging_level() -> LoggingLevel {
    *LOGGING_LEVEL.read().unwrap()
}

pub(crate) fn get_startup_time() -> Instant {
    unsafe { (*STARTUP_TIME.read().unwrap()).assume_init_read() }
}

pub(crate) fn setup_startup_time() {
    *STARTUP_TIME.write().unwrap() = MaybeUninit::new(Instant::now())
}

macro_rules! log {
    ($($arg:tt)*) => {{
        use crate::debug::{get_logging_level, get_startup_time, LoggingLevel};
        let message = format!($($arg)*);
        let now = get_startup_time().elapsed();
        match get_logging_level() {
            LoggingLevel::Console => println!("[{:>13.6}] [LOG] {}", now.as_secs_f32(), message),
            _ => ()
        }
    }};
}

macro_rules! error {
    ($($arg:tt)*) => {{
        use crate::debug::{get_logging_level, get_startup_time, LoggingLevel};
        let message = format!($($arg)*);
        let now = get_startup_time().elapsed();
        match get_logging_level() {
            LoggingLevel::Console => println!("[{:>13.6}] [ERROR] {}", now.as_secs_f32(), message),
            _ => ()
        }
    }};
}

macro_rules! fmt_size {
    ($size:expr) => {
        if $size >= 1024 * 1024 * 1024 {
            format!("{:.2}GB", $size as f32 / 1024. / 1024.)
        } else if $size >= 1024 * 1024 {
            format!("{:.2}MB", $size as f32 / 1024. / 1024.)
        } else if $size >= 1024 {
            format!("{:.2}KB", $size as f32 / 1024.)
        } else {
            format!("{}B", $size as u16)
        }
    };
}

use std::{mem::MaybeUninit, sync::RwLock, time::Instant};

pub(crate) use error;
pub(crate) use fmt_size;
pub(crate) use log;
