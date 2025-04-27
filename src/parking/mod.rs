//! MCS-CR lock implementations with thread parking support.
//!
//! This module provides implementations that **are not** `no_std` compatible.
//!
//! The [`raw`] module implements the same protocol as its root level counter
//! part and the same locking interfaces. The distinction is that this Mutex
//! implementations will transparently put the waiting threads to sleep under
//! some policy. Users are free to implement their own policies or pick sensible
//! ones under the [`park`] module. To define your own policy, users must
//! implement the [`Park`] trait.
//!
//! [`raw`]: crate::parking::raw
//! [`barging`]: crate::parking::barging
//! [`park`]: crate::parking::park
//! [`Park`]: crate::parking::park::Park

pub mod park;
pub mod raw;

mod parker;
