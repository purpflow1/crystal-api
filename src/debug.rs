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

use std::sync::RwLock;

pub(crate) use error;
pub(crate) use log;
