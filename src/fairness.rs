use crate::cfg::debug_abort;

/// A trait for types that define a fairness policy for passive tail promotion.
///
/// The type must be reachable through a `&'static Self`, since the access
/// and any form of state modification happens transparently from the crate's
/// users. This often achieved by storing the type at the thread local storage.
/// This is then main reason why this crate is not `no_std` compatible.
///
/// # Safety
///
/// All associated function implementations **must not** cause a thread exit,
/// such as envoking a uncaught [`core::panic!`] call, or any other operation
/// that will panic the thread. Exiting the thread will result in undefined
/// behiavior.
pub unsafe trait Fairness: 'static {
    /// Returns a shared static reference to `Self`.
    fn get() -> &'static Self;

    /// Determines whether to take a queue node from the tail of the passive set
    /// and set it as the lock holder's successor to ensure long-term fairness.
    fn should_promote_passive_tail(this: &'static Self) -> bool;
}

/// The actual implementation of this crate's `Fairness` types.
pub trait FairnessImpl: 'static {
    /// The actual `get` implementation.
    fn get() -> &'static Self;

    /// The actual `should_promote_passive_tail` implementation.
    fn should_promote_passive_tail(this: &'static Self) -> bool;
}

// SAFETY: Both `get` and `should_promote_passive_tail` function implementation
// are protected with a process abort (under test with unwind on panic
// configuration) in case any of them where to panic the thread.
#[doc(hidden)]
unsafe impl<F: FairnessImpl> Fairness for F {
    #[inline(always)]
    fn get() -> &'static Self {
        debug_abort::on_unwind(|| F::get())
    }

    #[inline(always)]
    fn should_promote_passive_tail(this: &'static Self) -> bool {
        debug_abort::on_unwind(|| F::should_promote_passive_tail(this))
    }
}
