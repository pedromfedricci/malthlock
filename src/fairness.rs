/// A trait for types that define a fairness policy for passive tail promotion.
///
/// The type must be reachable through a `&'static Self`, since the access
/// and any form of state modification happens transparently from the crate's
/// users. This often achieved by storing the type at the thread local storage.
/// This is then main reason why this crate is not `no_std` compatible.
pub trait Fairness: 'static {
    /// Returns a shared static reference to `Self`.
    fn get() -> &'static Self;

    /// Determines whether to take a queue node from the tail of the passive set
    /// and set it as the lock holder's successor to ensure long-term fairness.
    fn should_promote_passive_tail(this: &'static Self) -> bool;
}
