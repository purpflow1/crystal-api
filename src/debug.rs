/// Represents internal wrapper logging level
#[derive(Clone, Copy)]
pub enum LoggingLevel {
    /// Console output only
    Console,
    /// No logging
    None,
}

pub(crate) static LOGGING_LEVEL: RwLock<LoggingLevel> = RwLock::new(LoggingLevel::None);

/// Sets logging level
pub fn set_internal_logging_level(logging_level: LoggingLevel) {
    *LOGGING_LEVEL.write().unwrap() = logging_level
}

pub(crate) fn get_logging_level() -> LoggingLevel {
    *LOGGING_LEVEL.read().unwrap()
}

macro_rules! log {
    ($($arg:tt)*) => {{
        use crate::debug::{get_logging_level, LoggingLevel};
        let message = format!($($arg)*);
        match get_logging_level() {
            LoggingLevel::Console => println!("[LOG] {}", message),
            _ => ()
        }
    }};
}

macro_rules! error {
    ($($arg:tt)*) => {{
        use crate::debug::{get_logging_level, LoggingLevel};
        let message = format!($($arg)*);
        match get_logging_level() {
            LoggingLevel::Console => println!("[ERROR] {}", message),
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

use std::sync::RwLock;

pub(crate) use error;
pub(crate) use fmt_size;
pub(crate) use log;
