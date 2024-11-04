//! TODO: docs
//!
//! [`try_lock_then`]: Mutex::try_lock_then
//! [`try_lock_with_then`]: Mutex::try_lock_with_then
//! [`lock_then`]: Mutex::lock_then
//! [`lock_with_then`]: Mutex::lock_with_then
//! [`relax`]: crate::relax
//! [`Relax`]: crate::relax::Relax

mod mutex;
pub use mutex::{Mutex, MutexNode};

#[cfg(feature = "thread_local")]
#[cfg_attr(docsrs, doc(cfg(feature = "thread_local")))]
mod thread_local;
#[cfg(feature = "thread_local")]
pub use thread_local::LocalMutexNode;

/// A MCS-CR lock that implements a `spin` relax policy.
///
/// During lock contention, this lock spins while signaling the processor that
/// it is running a busy-wait spin-loop.
pub mod spins {
    use super::mutex;
    use crate::relax::Spin;

    /// A [`raw::Mutex`] that implements the [`Spin`] relax policy.
    ///
    /// # Example
    ///
    /// ```
    /// use malthlock::raw::{spins::Mutex, MutexNode};
    ///
    /// let mutex = Mutex::new(0);
    /// let mut node = MutexNode::new();
    /// let value = mutex.lock_with_then(&mut node, |data| *data);
    /// assert_eq!(value, 0);
    /// ```
    /// [`raw::Mutex`]: mutex::Mutex
    pub type Mutex<T> = mutex::Mutex<T, Spin>;

    /// A MCS-CR lock that implements a `spin with backoff` relax policy.
    ///
    /// During lock contention, this lock will perform exponential backoff
    /// while spinning, signaling the processor that it is running a busy-wait
    /// spin-loop.
    pub mod backoff {
        use super::mutex;
        use crate::relax::SpinBackoff;

        /// A [`raw::Mutex`] that implements the [`SpinBackoff`] relax policy.
        ///
        /// # Example
        ///
        /// ```
        /// use malthlock::raw::{spins::backoff::Mutex, MutexNode};
        ///
        /// let mutex = Mutex::new(0);
        /// let mut node = MutexNode::new();
        /// let value = mutex.lock_with_then(&mut node, |data| *data);
        /// assert_eq!(value, 0);
        /// ```
        /// [`raw::Mutex`]: mutex::Mutex
        pub type Mutex<T> = mutex::Mutex<T, SpinBackoff>;
    }
}

/// A MCS-CR lock that implements a `yield` relax policy.
///
/// During lock contention, this lock will yield the current time slice to the
/// OS scheduler.
#[cfg(any(feature = "yield", loom, test))]
#[cfg_attr(docsrs, doc(cfg(feature = "yield")))]
pub mod yields {
    use super::mutex;
    use crate::relax::Yield;

    /// A [`raw::Mutex`] that implements the [`Yield`] relax policy.
    ///
    /// # Example
    ///
    /// ```
    /// use malthlock::raw::{yields::Mutex, MutexNode};
    ///
    /// let mutex = Mutex::new(0);
    /// let mut node = MutexNode::new();
    /// let value = mutex.lock_with_then(&mut node, |data| *data);
    /// assert_eq!(value, 0);
    /// ```
    /// [`raw::Mutex`]: mutex::Mutex
    pub type Mutex<T> = mutex::Mutex<T, Yield>;

    /// A MCS-CR lock that implements a `yield with backoff` relax policy.
    ///
    /// During lock contention, this lock will perform exponential backoff while
    /// spinning, up to a threshold, then yields back to the OS scheduler.
    pub mod backoff {
        use super::mutex;
        use crate::relax::YieldBackoff;

        /// A [`raw::Mutex`] that implements the [`YieldBackoff`] relax policy.
        ///
        /// # Example
        ///
        /// ```
        /// use malthlock::raw::{yields::backoff::Mutex, MutexNode};
        ///
        /// let mutex = Mutex::new(0);
        /// let mut node = MutexNode::new();
        /// let value = mutex.lock_with_then(&mut node, |data| *data);
        /// assert_eq!(value, 0);
        /// ```
        /// [`raw::Mutex`]: mutex::Mutex
        pub type Mutex<T> = mutex::Mutex<T, YieldBackoff>;
    }
}

/// A MCS-CR lock that implements a `loop` relax policy.
///
/// During lock contention, this lock will rapidly spin without telling the CPU
/// to do any power down.
pub mod loops {
    use super::mutex;
    use crate::relax::Loop;

    /// A [`raw::Mutex`] that implements the [`Loop`] relax policy.
    ///
    /// # Example
    ///
    /// ```
    /// use malthlock::raw::{loops::Mutex, MutexNode};
    ///
    /// let mutex = Mutex::new(0);
    /// let mut node = MutexNode::new();
    /// let value = mutex.lock_with_then(&mut node, |data| *data);
    /// assert_eq!(value, 0);
    /// ```
    /// [`raw::Mutex`]: mutex::Mutex
    pub type Mutex<T> = mutex::Mutex<T, Loop>;
}
