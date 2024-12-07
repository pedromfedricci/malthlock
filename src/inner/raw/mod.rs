use core::fmt::{self, Debug, Display, Formatter};
use core::marker::PhantomData;
use core::mem::MaybeUninit;
use core::ptr;
use core::sync::atomic::Ordering::{AcqRel, Acquire, Relaxed, Release};

use crate::cfg::atomic::{fence, AtomicPtr, AtomicPtrNull, UnsyncLoad};
use crate::cfg::cell::{Cell, CellNullMut, UnsafeCell, UnsafeCellOptionWith, UnsafeCellWith};
use crate::fairness::Fairness;
use crate::lock::{Lock, Wait};
use crate::relax::Relax;

#[cfg(not(all(loom, test)))]
use core::ops::{Deref, DerefMut};

#[cfg(feature = "thread_local")]
mod thread_local;
#[cfg(feature = "thread_local")]
pub use thread_local::LocalMutexNode;

/// The inner type of [`MutexNode`], which is known to be in initialized state.
#[derive(Debug)]
pub struct MutexNodeInit<L> {
    next: AtomicPtr<Self>,
    prev: Cell<*mut Self>,
    lock: L,
}

impl<L> MutexNodeInit<L> {
    /// Returns a raw mutable pointer of this node.
    const fn as_ptr(&self) -> *mut Self {
        (self as *const Self).cast_mut()
    }

    /// A relaxed loop that returns a pointer to the successor once it finishes
    /// linking with the current thread.
    ///
    /// The atomic load operation called inside the loop is relaxed, but the
    /// returned node pointer is synchronized through a acquire fence.
    fn wait_next_acquire<R: Relax>(&self) -> *mut Self {
        let mut relax = R::new();
        let next = loop {
            let ptr = self.next.load(Relaxed);
            let true = ptr.is_null() else { break ptr };
            relax.relax();
        };
        fence(Acquire);
        next
    }

    /// Updates first node's `next` pointer with second node's value.
    ///
    /// # Safety
    ///
    /// Both pointers are required to be non-null and aligned, and the current
    /// thread must have exclusive access over them.
    unsafe fn link_next(this: *mut Self, node: *mut Self) {
        // SAFETY: Caller guaranteed that `this` is valid and that the current
        // thread has exclusive access over it.
        *unsafe { &mut (*this).next } = AtomicPtr::new(node);
    }

    /// Updates first node's `prev` pointer with second node's value.
    ///
    /// # Safety
    ///
    /// Both pointers are required to be non-null and aligned, and the current
    /// thread must have exclusive access over them.
    unsafe fn link_prev(this: *mut Self, node: *mut Self) {
        // SAFETY: Caller guaranteed that `this` is valid and that the current
        // thread has exclusive access over it.
        unsafe { &(*this).prev }.set(node);
    }

    /// Sets node's `next` and `prev` pointers to `null` and return it.
    ///
    /// # Safety
    ///
    /// Pointer is required to be non-null and aligned, and the current thread
    /// must have exclusive access over it.
    unsafe fn unlink(this: *mut Self) -> *mut Self {
        // SAFETY: Caller guaranteed that `this` is valid and that the current
        // thread has exclusive access over it.
        unsafe {
            Self::unlink_next(this);
            Self::unlink_prev(this);
        }
        this
    }

    /// Sets node's `next` pointer to `null`.
    ///
    /// # Safety
    ///
    /// Pointer is required to be non-null and aligned, and the current thread
    /// must have exclusive access over it.
    unsafe fn unlink_next(this: *mut Self) {
        // SAFETY: Caller guaranteed that `this` is valid and that the current
        // thread has exclusive access over it.
        *unsafe { &mut (*this).next } = AtomicPtr::null_mut();
    }

    /// Sets node's `prev` pointer to `null`.
    ///
    /// # Safety
    ///
    /// Pointer is required to be non-null and aligned, and the current thread
    /// must have exclusive access over it.
    unsafe fn unlink_prev(this: *mut Self) {
        // SAFETY: Caller guaranteed that `this` is valid and that the current
        // thread has exclusive access over it.
        unsafe { &(*this).prev }.set_null();
    }

    /// Gets the target node's `next` pointer.
    ///
    /// # Safety
    ///
    /// Pointer is required to be non-null and aligned, and the current thread
    /// must have exclusive access over it.
    unsafe fn get_next(this: *mut Self) -> *mut Self {
        // SAFETY: Caller guaranteed that `this` is valid and that the current
        // thread has exclusive access over it.
        unsafe { &mut (*this).next }.load_unsynced()
    }

    /// Gets the target node's `prev` pointer.
    ///
    /// # Safety
    ///
    /// Pointer is required to be non-null and aligned, and the current thread
    /// must have exclusive access over it.
    unsafe fn get_prev(this: *mut Self) -> *mut Self {
        // SAFETY: Caller guaranteed that `this` is valid and that the current
        // thread has exclusive access over it.
        unsafe { &(*this).prev }.get()
    }
}

impl<L: Lock> MutexNodeInit<L> {
    /// Creates a new, locked and core based node (const).
    #[cfg(not(all(loom, test)))]
    const fn locked() -> Self {
        let next = AtomicPtr::NULL_MUT;
        let prev = Cell::NULL_MUT;
        let lock = Lock::LOCKED;
        Self { next, prev, lock }
    }

    /// Creates a new, locked and loom based node (non-const).
    #[cfg(all(loom, test))]
    #[cfg(not(tarpaulin_include))]
    fn locked() -> Self {
        let next = AtomicPtr::null_mut();
        let prev = Cell::null_mut();
        let lock = Lock::locked();
        Self { next, prev, lock }
    }
}

/// A locally-accessible record for forming the waiting queue.
///
/// The inner state is never dropped, only overwritten. This is desirable and
/// well suited for our use cases, since all `L` types used are only composed
/// of `no drop glue` types (eg. atomic types).
///
/// `L` must fail [`core::mem::needs_drop`] check, else `L` will leak.
#[derive(Debug)]
#[repr(transparent)]
pub struct MutexNode<L> {
    inner: MaybeUninit<MutexNodeInit<L>>,
}

impl<L> MutexNode<L> {
    /// Creates new `MutexNode` instance.
    pub const fn new() -> Self {
        Self { inner: MaybeUninit::uninit() }
    }
}

impl<L: Lock> MutexNode<L> {
    /// Initializes this node's inner state, returning a shared reference
    /// pointing to it.
    fn initialize(&mut self) -> &MutexNodeInit<L> {
        self.inner.write(MutexNodeInit::locked())
    }
}

#[cfg(not(tarpaulin_include))]
impl<L> Default for MutexNode<L> {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

/// The passive set keeps track of "cold" queue nodes that were excised from
/// the main queue.
///
/// The contention reduction protocol directly edits the MCS chain to shift
/// threads back and forth between the main chain and the explicit list of
/// passivated threads.
///
/// The element at the head of passive list is the most recently arrived member
/// of the passive set.
///
/// The passive set is exclusively edited at unlock time and, during that time,
/// only the unlocking thread must have access to the passive set, concurrent
/// access is not permited.
struct PassiveSet<L> {
    head: Cell<*mut MutexNodeInit<L>>,
    tail: Cell<*mut MutexNodeInit<L>>,
}

// SAFETY: The passive set is not a self-referential struct. Its pointers are
// only accessed during the unlock operation, where no concurrent access over
// the passive set is performed. The unlocking thread is the only accessor.
unsafe impl<L> Send for PassiveSet<L> {}

impl<L> PassiveSet<L> {
    #[cfg(not(all(loom, test)))]
    /// Creates a new, core based passive set (const).
    const fn new() -> Self {
        let head = Cell::NULL_MUT;
        let tail = Cell::NULL_MUT;
        Self { head, tail }
    }

    /// Creates a new, loom based passive set (non-const).
    #[cfg(all(loom, test))]
    fn new() -> Self {
        let head = Cell::null_mut();
        let tail = Cell::null_mut();
        Self { head, tail }
    }

    /// Pushes target node at the front of the passive queue.
    ///
    /// # Safety
    ///
    /// Pointer is required to be non-null and aligned. The current thread
    /// must have exclusive access over the passive set and the target node.
    unsafe fn push_front(&self, node: *mut MutexNodeInit<L>) {
        let head = self.head.get();
        // SAFETY: Caller guaranteed that `node` is a non-null and aligned
        // pointer, and that this thread has exclusive access to its value.
        unsafe { MutexNodeInit::link_next(node, head) };
        unsafe { MutexNodeInit::unlink_prev(node) };
        self.head.set(node);
        // SAFETY: Already verified that `head` pointer is not null.
        (!head.is_null()).then(|| unsafe { MutexNodeInit::link_prev(head, node) });
        self.tail.get().is_null().then(|| self.tail.set(node));
    }

    /// Removes the last node from the back of the passive set.
    ///
    /// # Safety
    ///
    /// The current thread must have exclusive access over the passive set.
    unsafe fn pop_back(&self) -> *mut MutexNodeInit<L> {
        let tail = self.tail.get();
        let false = tail.is_null() else { return tail };
        // SAFETY: Already verified that `tail` pointer is not null and caller
        // guaranteed that the current thread has exclusive access over the
        // passive set.
        self.tail.set(unsafe { MutexNodeInit::get_prev(tail) });
        if self.tail.get().is_null() {
            self.head.set_null();
        } else {
            // SAFETY: Caller guaranteed that the current thread has exclusive
            // access over the passive set.
            unsafe { self.unlink_tail_next() };
        }
        // SAFETY: Already verified that `tail` pointer is not null and caller
        // guaranteed that the current thread has exclusive access over the
        // passive set.
        unsafe { MutexNodeInit::unlink(tail) }
    }

    /// Removes the first node from the front of the passive set.
    ///
    /// # Safety
    ///
    /// The current thread must have exclusive access over the passive set.
    unsafe fn pop_front(&self) -> *mut MutexNodeInit<L> {
        let head = self.head.get();
        let false = head.is_null() else { return head };
        // SAFETY: Already verified that `head` pointer is not null and caller
        // guaranteed that the current thread has exclusive access over the
        // passive set.
        self.head.set(unsafe { MutexNodeInit::get_next(head) });
        if self.head.get().is_null() {
            self.tail.set_null();
        } else {
            // SAFETY: Caller guaranteed that the current thread has exclusive
            // access over the passive set.
            unsafe { self.unlink_head_prev() };
        }
        // SAFETY: Already verified that `head` pointer is not null and caller
        // guaranteed that the current thread has exclusive access over the
        // passive set.
        unsafe { MutexNodeInit::unlink(head) }
    }

    /// Sets `tail`'s next pointer to `null`.
    ///
    /// # Safety
    ///
    /// Self's `tail` must be a non-null, well aligned pointer and the current
    /// thread must have exclusive access to it.
    unsafe fn unlink_tail_next(&self) {
        let tail = self.tail.get();
        // SAFETY: Caller guaranteed that `tail` is valid and that the current
        // thread has exclusive access over it.
        unsafe { MutexNodeInit::unlink_next(tail) };
    }

    /// Sets `head`'s previous pointer to `null`.
    ///
    /// # Safety
    ///
    /// Self's `head` must be a non-null, well aligned pointer and the current
    /// thread must have exclusive access to it.
    unsafe fn unlink_head_prev(&self) {
        let head = self.head.get();
        // SAFETY: Caller guaranteed that `head` is valid and that the current
        // thread has exclusive access over it.
        unsafe { MutexNodeInit::unlink_prev(head) };
    }
}

/// A mutual exclusion primitive implementing the MCS-CR lock protocol, useful
/// for protecting shared data.
pub struct Mutex<T: ?Sized, L, W, F> {
    tail: AtomicPtr<MutexNodeInit<L>>,
    passive_set: PassiveSet<L>,
    marker: PhantomData<(W, F)>,
    data: UnsafeCell<T>,
}

// Same unsafe impls as `std::sync::Mutex`.
unsafe impl<T: ?Sized + Send, L, W, F> Send for Mutex<T, L, W, F> {}
unsafe impl<T: ?Sized + Send, L, W, F> Sync for Mutex<T, L, W, F> {}

impl<T, L, W, F> Mutex<T, L, W, F> {
    /// Creates a new, unlocked and core based mutex (const).
    #[cfg(not(all(loom, test)))]
    pub const fn new(value: T) -> Self {
        let tail = AtomicPtr::NULL_MUT;
        let passive_set = PassiveSet::new();
        let data = UnsafeCell::new(value);
        Self { tail, data, passive_set, marker: PhantomData }
    }

    /// Creates a new, unlocked and loom based mutex (non-const).
    #[cfg(all(loom, test))]
    #[cfg(not(tarpaulin_include))]
    pub fn new(value: T) -> Self {
        let tail = AtomicPtr::null_mut();
        let passive_set = PassiveSet::new();
        let data = UnsafeCell::new(value);
        Self { tail, data, passive_set, marker: PhantomData }
    }

    /// Consumes this mutex, returning the underlying data.
    pub fn into_inner(self) -> T {
        self.data.into_inner()
    }
}

impl<T: ?Sized, L: Lock, W: Wait, F: Fairness> Mutex<T, L, W, F> {
    /// Attempts to acquire this mutex without blocking the thread.
    ///
    /// # Safety
    ///
    /// The returned guard instance **must** be dropped, that is, it **must not**
    /// be "forgotten" (e.g. `core::mem::forget`), or being targeted by any
    /// other operation that would prevent it from executing its `drop` call.
    unsafe fn try_lock_with<'a>(&'a self, n: &'a mut MutexNode<L>) -> OptionGuard<'a, T, L, W, F> {
        let node = n.initialize();
        self.tail
            .compare_exchange(ptr::null_mut(), node.as_ptr(), AcqRel, Relaxed)
            .map(|_| MutexGuard::new(self, node))
            .ok()
    }

    /// Acquires this mutex, blocking the current thread until it is able to do so.
    ///
    /// # Safety
    ///
    /// The returned guard instance **must** be dropped, that is, it **must not**
    /// be "forgotten" (e.g. `core::mem::forget`), or being targeted of any
    /// other operation that would prevent it from executing its `drop` call.
    unsafe fn lock_with<'a>(&'a self, n: &'a mut MutexNode<L>) -> MutexGuard<'a, T, L, W, F> {
        let node = n.initialize();
        let pred = self.tail.swap(node.as_ptr(), AcqRel);
        // If we have a predecessor, complete the link so it will notify us.
        if !pred.is_null() {
            // SAFETY: Already verified that our predecessor is not null.
            unsafe { &(*pred).next }.store(node.as_ptr(), Release);
            // Verify the lock hand-off, while applying some waiting policy.
            node.lock.wait_lock_relaxed::<W>();
            fence(Acquire);
        }
        MutexGuard::new(self, node)
    }

    /// Unlocks this mutex.
    ///
    /// # Safety
    ///
    /// The current thread must be the unlocking thread, that is, it must have
    /// exclusive access over the passive set.
    unsafe fn unlock_with(&self, head: &MutexNodeInit<L>) {
        if should_promote_passive_tail::<F>() {
            // SAFETY: Caller guaranteed that the current thread has exclusive
            // access over the passive set.
            let promoted = unsafe { self.promote_passive_tail(head) };
            let false = promoted else { return };
        }
        let next = head.next.load(Acquire);
        if next.is_null() {
            // SAFETY: Caller guaranteed that the current thread has exclusive
            // access over the passive set.
            unsafe { self.promote_passive_head(head) };
        } else if next != self.tail.load(Acquire) {
            // SAFETY: Already verified that `next` pointer is not null and
            // caller guaranteed that the current thread has exclusive access
            // over the passive set.
            unsafe { self.demote_active_head_next(head, &*next) };
        } else {
            // SAFETY: Already verified that `next` pointer is not null.
            unsafe { &(*next).lock }.notify_release();
        }
    }

    /// Excise the head's successor from the active list and transfer it to the
    /// passive set.
    ///
    /// # Safety
    ///
    /// The current thread must be the unlocking thread, that is, it must have
    /// exclusive access over the passive set.
    unsafe fn demote_active_head_next(&self, head: &MutexNodeInit<L>, next: &MutexNodeInit<L>) {
        let new_next = next.wait_next_acquire::<W::UnlockRelax>();
        // SAFETY: Caller guaranteed that the current thread is the unlocking
        // thread and therefore it has exclusive access over the passive set.
        unsafe { self.passive_set.push_front(next.as_ptr()) };
        // SAFETY: Head's successor has already finished linking, therefore
        // this thread has exclusive access over head's `next` pointer.
        unsafe { MutexNodeInit::link_next(head.as_ptr(), new_next) };
        // SAFETY: Already verified that head's new successor is not null.
        unsafe { &(*new_next).lock }.notify_release();
    }

    /// Selects the tail of the passive set (if any) as the successor of the
    /// lock holder to ensure long-term fairness.
    ///
    /// If the passive set is empty, returns `false`, indicating that the tail
    /// was not transfered as the lock holder's successor. On transfer success,
    /// returns `true`.
    ///
    /// # Safety
    ///
    /// The current thread must be the unlocking thread, that is, it must have
    /// exclusive access over the passive set.
    unsafe fn promote_passive_tail(&self, head: &MutexNodeInit<L>) -> bool {
        // SAFETY: Caller guaranteed that the current thread is the unlocking
        // thread and therefore it has exclusive access over the passive set.
        let passive_tail = unsafe { self.passive_set.pop_back() };
        let false = passive_tail.is_null() else { return false };
        // SAFETY: Already verified that `passive_tail` pointer is not null and
        // caller guaranteed that the current thread has exclusive access over
        // the passive set.
        unsafe {
            self.link_active_head_next(head, passive_tail);
            (*passive_tail).lock.notify_release();
        }
        true
    }

    /// Promotes the head of the passive set (if any), transfering the node
    /// into the active queue as the lock holder's successor, assuming that
    /// `head` is the only node in the queue.
    ///
    /// # Safety
    ///
    /// The current thread must be the unlocking thread, that is, it must have
    /// exclusive access over the passive set.
    unsafe fn promote_passive_head(&self, head: &MutexNodeInit<L>) {
        // SAFETY: Caller guaranteed that the current thread is the unlocking
        // thread and therefore it has exclusive access over the passive set.
        let passive_head = unsafe { self.passive_set.pop_front() };
        if passive_head.is_null() {
            let false = self.try_unlock_release(head.as_ptr()) else { return };
            let next = head.wait_next_acquire::<W::UnlockRelax>();
            // SAFETY: Successor has already finished linking, therefore `next`
            // pointer is not null.
            unsafe { &(*next).lock }.notify_release();
        } else {
            // SAFETY: Already verified that `passive_head` pointer is not null
            // and caller guaranteed that the current thread has exclusive
            // access over the passive set.
            unsafe {
                self.link_active_head_next(head, passive_head);
                (*passive_head).lock.notify_release();
            }
        }
    }

    /// Insert a target node as the lock holder's successor.
    ///
    /// # Safety
    ///
    /// Target node must be a non-null and aligned pointer, and the current
    /// thread must have exclusive access over it throughout this call.
    unsafe fn link_active_head_next(&self, head: &MutexNodeInit<L>, node: *mut MutexNodeInit<L>) {
        let tail = self.tail.compare_exchange(head.as_ptr(), node, AcqRel, Relaxed);
        if let Ok(head) = tail {
            // SAFETY: On success, `head` has no pending successors and it is
            // no longer the queue's tail, therefore this thread has exclusive
            // acceess over its `next` pointer.
            unsafe { MutexNodeInit::link_next(head, node) };
        } else {
            let next = head.wait_next_acquire::<W::UnlockRelax>();
            // SAFETY: Caller guaranteed that `node` is a non-null and aligned
            // pointer and that this thread has exclusive access over it.
            unsafe { MutexNodeInit::link_next(node, next) };
            // SAFETY: The `head` is not the queue's tail and its successor has
            // already finished linking, therefore this thread has exclusive
            // access over its `next` pointer.
            unsafe { MutexNodeInit::link_next(head.as_ptr(), node) };
            fence(Acquire);
        }
    }
}

impl<T: ?Sized, L, W, F> Mutex<T, L, W, F> {
    /// Returns `true` if the lock is currently held.
    ///
    /// This function does not guarantee strong ordering, only atomicity.
    pub fn is_locked(&self) -> bool {
        !self.tail.load(Relaxed).is_null()
    }

    /// Returns a mutable reference to the underlying data.
    #[cfg(not(all(loom, test)))]
    pub fn get_mut(&mut self) -> &mut T {
        // SAFETY: We hold exclusive access to the Mutex data.
        unsafe { &mut *self.data.get() }
    }

    /// Unlocks the lock if the candidate node is the queue's tail.
    fn try_unlock_release(&self, node: *mut MutexNodeInit<L>) -> bool {
        self.tail.compare_exchange(node, ptr::null_mut(), Release, Relaxed).is_ok()
    }
}

impl<T: ?Sized, L: Lock, W: Wait, F: Fairness> Mutex<T, L, W, F> {
    /// Attempts to acquire this mutex and then runs a closure against the
    /// protected data.
    ///
    /// This function does not block.
    pub fn try_lock_with_then<Fn, Ret>(&self, node: &mut MutexNode<L>, f: Fn) -> Ret
    where
        Fn: FnOnce(Option<&mut T>) -> Ret,
    {
        // SAFETY: The guard's `drop` call is executed within this scope.
        unsafe { self.try_lock_with(node) }.as_deref_mut_with_mut(f)
    }

    /// Acquires this mutex and then runs the closure against the protected data.
    ///
    /// This function will block if the lock is unavailable.
    pub fn lock_with_then<Fn, Ret>(&self, node: &mut MutexNode<L>, f: Fn) -> Ret
    where
        Fn: FnOnce(&mut T) -> Ret,
    {
        // SAFETY: The guard's `drop` call is executed within this scope.
        unsafe { self.lock_with(node) }.with_mut(f)
    }
}

impl<T: ?Sized + Debug, L: Lock, W: Wait, F: Fairness> Debug for Mutex<T, L, W, F> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let mut node = MutexNode::new();
        let mut d = f.debug_struct("Mutex");
        self.try_lock_with_then(&mut node, |data| match data {
            Some(data) => d.field("data", &data),
            None => d.field("data", &format_args!("<locked>")),
        });
        d.finish()
    }
}

/// Determines whether to take a queue node from the tail of the passive set
/// and set it as the lock holder's successor to ensure long-term fairness.
fn should_promote_passive_tail<F: Fairness>() -> bool {
    F::should_promote_passive_tail(F::get())
}

/// Short alias for a `Option` wrapped `MutexGuard`;
type OptionGuard<'a, T, L, W, F> = Option<MutexGuard<'a, T, L, W, F>>;

/// An RAII implementation of a "scoped lock" of a mutex. When this structure is
/// dropped (falls out of scope), the lock will be unlocked.
#[must_use = "if unused the Mutex will immediately unlock"]
pub struct MutexGuard<'a, T: ?Sized, L: Lock, W: Wait, F: Fairness> {
    lock: &'a Mutex<T, L, W, F>,
    head: &'a MutexNodeInit<L>,
}

// Same unsafe Sync impl as `std::sync::MutexGuard`.
unsafe impl<T: ?Sized + Sync, L: Lock, W: Wait, F: Fairness> Sync for MutexGuard<'_, T, L, W, F> {}

impl<'a, T: ?Sized, L: Lock, W: Wait, F: Fairness> MutexGuard<'a, T, L, W, F> {
    /// Creates a new `MutexGuard` instance.
    const fn new(lock: &'a Mutex<T, L, W, F>, head: &'a MutexNodeInit<L>) -> Self {
        Self { lock, head }
    }

    /// Runs `f` against a shared reference pointing to the underlying data.
    #[cfg(not(tarpaulin_include))]
    fn with<Fn, Ret>(&self, f: Fn) -> Ret
    where
        Fn: FnOnce(&T) -> Ret,
    {
        // SAFETY: A guard instance holds the lock locked.
        unsafe { self.lock.data.with_unchecked(f) }
    }

    /// Runs `f` against a mutable reference pointing to the underlying data.
    fn with_mut<Fn, Ret>(&mut self, f: Fn) -> Ret
    where
        Fn: FnOnce(&mut T) -> Ret,
    {
        // SAFETY: A guard instance holds the lock locked.
        unsafe { self.lock.data.with_mut_unchecked(f) }
    }
}

/// A trait that converts a `&mut Self` to a `Option<&mut Self::Target>` and
/// then runs closures against it.
trait AsDerefMutWithMut {
    type Target: ?Sized;

    /// Converts `&mut Self` to `Option<&mut Self::Target>` and then runs `f`
    /// against it.
    fn as_deref_mut_with_mut<F, Ret>(&mut self, f: F) -> Ret
    where
        F: FnOnce(Option<&mut Self::Target>) -> Ret;
}

impl<T: ?Sized, L: Lock, W: Wait, F: Fairness> AsDerefMutWithMut for OptionGuard<'_, T, L, W, F> {
    type Target = T;

    fn as_deref_mut_with_mut<Fn, Ret>(&mut self, f: Fn) -> Ret
    where
        Fn: FnOnce(Option<&mut Self::Target>) -> Ret,
    {
        let data = self.as_ref().map(|guard| &guard.lock.data);
        // SAFETY: A guard instance holds the lock locked.
        unsafe { data.as_deref_with_mut_unchecked(f) }
    }
}

#[cfg(not(tarpaulin_include))]
impl<T: ?Sized + Debug, L: Lock, W: Wait, F: Fairness> Debug for MutexGuard<'_, T, L, W, F> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        self.with(|data| data.fmt(f))
    }
}

#[cfg(not(tarpaulin_include))]
impl<T: ?Sized + Display, L: Lock, W: Wait, F: Fairness> Display for MutexGuard<'_, T, L, W, F> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        self.with(|data| data.fmt(f))
    }
}

#[cfg(not(all(loom, test)))]
#[cfg(not(tarpaulin_include))]
impl<T: ?Sized, L: Lock, W: Wait, F: Fairness> Deref for MutexGuard<'_, T, L, W, F> {
    type Target = T;

    /// Dereferences the guard to access the underlying data.
    fn deref(&self) -> &T {
        // SAFETY: A guard instance holds the lock locked.
        unsafe { &*self.lock.data.get() }
    }
}

#[cfg(not(all(loom, test)))]
#[cfg(not(tarpaulin_include))]
impl<T: ?Sized, L: Lock, W: Wait, F: Fairness> DerefMut for MutexGuard<'_, T, L, W, F> {
    /// Mutably dereferences the guard to access the underlying data.
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY: A guard instance holds the lock locked.
        unsafe { &mut *self.lock.data.get() }
    }
}

impl<T: ?Sized, L: Lock, W: Wait, F: Fairness> Drop for MutexGuard<'_, T, L, W, F> {
    fn drop(&mut self) {
        // SAFETY: At most one guard drop call may be running at any given time
        // for its associated mutex. In that period, the unlocking thread is
        // the sole accessor of the mutex's passive set.
        unsafe { self.lock.unlock_with(self.head) }
    }
}
