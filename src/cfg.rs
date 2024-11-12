pub mod atomic {
    pub use sealed::{AtomicPtrNull, UnsyncLoad};

    #[cfg(not(all(loom, test)))]
    pub use core::sync::atomic::{fence, AtomicBool, AtomicPtr};

    #[cfg(all(loom, test))]
    pub use loom::sync::atomic::{fence, AtomicBool, AtomicPtr};

    impl<T> UnsyncLoad for AtomicPtr<T> {
        type Target = T;

        #[cfg(not(all(loom, test)))]
        fn load_unsynced(&mut self) -> *mut Self::Target {
            *self.get_mut()
        }

        #[cfg(all(loom, test))]
        #[cfg(not(tarpaulin_include))]
        fn load_unsynced(&mut self) -> *mut Self::Target {
            self.with_mut(|ptr| *ptr)
        }
    }

    impl<T> AtomicPtrNull for AtomicPtr<T> {
        type Target = T;

        #[rustfmt::skip]
        #[cfg(not(all(loom, test)))]
        const NULL_MUT: AtomicPtr<Self::Target> = {
            Self::new(core::ptr::null_mut())
        };

        fn null_mut() -> AtomicPtr<Self::Target> {
            Self::new(core::ptr::null_mut())
        }
    }

    mod sealed {
        use super::AtomicPtr;

        /// A trait that extends [`AtomicPtr`] so that it will allow loading the
        /// the raw pointer without any synchronization.
        ///
        pub trait UnsyncLoad {
            /// The type of the pointed to value.
            type Target;

            /// Load the value without any synchronization.
            fn load_unsynced(&mut self) -> *mut Self::Target;
        }

        /// A trait that extends [`AtomicPtr`] to allow creating `null` values.
        pub trait AtomicPtrNull {
            /// The type of the data pointed to.
            type Target;

            /// A compiler time evaluable [`AtomicPtr`] poiting to `null`.
            #[cfg(not(all(loom, test)))]
            #[allow(clippy::declare_interior_mutable_const)]
            const NULL_MUT: AtomicPtr<Self::Target>;

            /// Returns a [`AtomicPtr`] instance poiting to `null` (non-const).
            fn null_mut() -> AtomicPtr<Self::Target>;
        }
    }
}

pub mod cell {
    pub use sealed::{CellNullMut, UnsafeCellOptionWith, UnsafeCellWith};

    #[cfg(not(all(loom, test)))]
    pub use core::cell::{Cell, UnsafeCell};

    #[cfg(all(loom, test))]
    pub use loom::cell::{Cell, UnsafeCell};

    impl<T: ?Sized> UnsafeCellWith for UnsafeCell<T> {
        type Target = T;

        #[cfg(not(all(loom, test)))]
        unsafe fn with_unchecked<F, Ret>(&self, f: F) -> Ret
        where
            F: FnOnce(&Self::Target) -> Ret,
        {
            // SAFETY: Caller guaranteed that there are no mutable aliases.
            f(unsafe { &*self.get() })
        }

        #[cfg(all(loom, test))]
        #[cfg(not(tarpaulin_include))]
        unsafe fn with_unchecked<F, Ret>(&self, f: F) -> Ret
        where
            F: FnOnce(&Self::Target) -> Ret,
        {
            // SAFETY: Caller guaranteed that there are no mutable aliases.
            self.with(|ptr| f(unsafe { &*ptr }))
        }

        #[cfg(not(all(loom, test)))]
        unsafe fn with_mut_unchecked<F, Ret>(&self, f: F) -> Ret
        where
            F: FnOnce(&mut Self::Target) -> Ret,
        {
            // SAFETY: Caller guaranteed that there are no other aliases.
            f(unsafe { &mut *self.get() })
        }

        #[cfg(all(loom, test))]
        #[cfg(not(tarpaulin_include))]
        unsafe fn with_mut_unchecked<F, Ret>(&self, f: F) -> Ret
        where
            F: FnOnce(&mut Self::Target) -> Ret,
        {
            // SAFETY: Caller guaranteed that there are no other aliases.
            self.with_mut(|ptr| f(unsafe { &mut *ptr }))
        }
    }

    impl<T: ?Sized> UnsafeCellOptionWith for Option<&UnsafeCell<T>> {
        type Target = T;

        #[cfg(not(all(loom, test)))]
        unsafe fn as_deref_with_mut_unchecked<F, Ret>(&self, f: F) -> Ret
        where
            F: FnOnce(Option<&mut T>) -> Ret,
        {
            let ptr = self.map(UnsafeCell::get);
            // SAFETY: Caller guaranteed that there are no mutable aliases.
            f(ptr.map(|ptr| unsafe { &mut *ptr }))
        }

        #[cfg(all(loom, test))]
        #[cfg(not(tarpaulin_include))]
        unsafe fn as_deref_with_mut_unchecked<F, Ret>(&self, f: F) -> Ret
        where
            F: FnOnce(Option<&mut T>) -> Ret,
        {
            let ptr = self.map(UnsafeCell::get_mut);
            // SAFETY: Caller guaranteed that there are no mutable aliases.
            f(ptr.as_ref().map(|ptr| unsafe { ptr.deref() }))
        }
    }

    impl<T> CellNullMut for Cell<*mut T> {
        type Target = T;

        #[rustfmt::skip]
        #[cfg(not(all(loom, test)))]
        const NULL_MUT: Cell<*mut Self::Target> = {
            Self::new(core::ptr::null_mut())
        };

        #[cfg(all(loom, test))]
        fn null_mut() -> Cell<*mut Self::Target> {
            Self::new(core::ptr::null_mut())
        }

        fn set_null(&self) {
            self.set(core::ptr::null_mut());
        }
    }

    mod sealed {
        use super::Cell;

        /// A trait that extends [`UnsafeCell`] to allow running closures against
        /// its underlying data.
        pub trait UnsafeCellWith {
            /// The type of the underlying data.
            type Target: ?Sized;

            /// Runs `f` against a shared reference borrowed from a [`UnsafeCell`].
            ///
            /// # Safety
            ///
            /// Caller must guarantee there are no mutable aliases to the
            /// underlying data.
            unsafe fn with_unchecked<F, Ret>(&self, f: F) -> Ret
            where
                F: FnOnce(&Self::Target) -> Ret;

            /// Runs `f` against a exclusive reference borrowed from a [`UnsafeCell`].
            ///
            /// # Safety
            ///
            /// Caller must guarantee there are no other aliases to the
            /// underlying data.
            unsafe fn with_mut_unchecked<F, Ret>(&self, f: F) -> Ret
            where
                F: FnOnce(&mut Self::Target) -> Ret;
        }

        /// A trait that extends `Option<&UnsafeCell>` to allow running closures
        /// against its underlying data.
        pub trait UnsafeCellOptionWith {
            type Target: ?Sized;

            /// Runs `f` against a mutable reference borrowed from a
            /// [`Option<&UnsafeCell>`].
            ///
            /// # Safety
            ///
            /// Caller must guarantee there are no mutable aliases to the
            /// underlying data.
            unsafe fn as_deref_with_mut_unchecked<F, Ret>(&self, f: F) -> Ret
            where
                F: FnOnce(Option<&mut Self::Target>) -> Ret;
        }

        /// A trait that extends [`Cell`] to allow creating `null` values.
        pub trait CellNullMut {
            /// The type of the data inner pointer points to.
            type Target;

            /// A compiler time evaluable [`Cell`] holding a `null` pointer.
            #[cfg(not(all(loom, test)))]
            #[allow(clippy::declare_interior_mutable_const)]
            const NULL_MUT: Cell<*mut Self::Target>;

            /// Returns a Loom based [`Cell`] holding a `null` pointer (non-const).
            #[cfg(all(loom, test))]
            fn null_mut() -> Cell<*mut Self::Target>;

            /// Sets the inner value to `null` through a shared reference.
            fn set_null(&self);
        }
    }
}

pub mod debug_abort {
    #[cfg(all(test, panic = "unwind"))]
    use std::panic::Location;

    /// Runs the closure, aborting the process if a unwinding panic occurs.
    #[cfg(all(test, panic = "unwind"))]
    #[track_caller]
    pub fn on_unwind<T, F: FnOnce() -> T>(may_unwind: F) -> T {
        let location = Location::caller();
        let abort = DebugAbort { location };
        let value = may_unwind();
        core::mem::forget(abort);
        value
    }

    /// Runs the closure, without aborting the process if a unwinding panic
    /// occurs.
    #[cfg(not(all(test, panic = "unwind")))]
    #[cfg(not(tarpaulin_include))]
    pub fn on_unwind<T, F: FnOnce() -> T>(may_unwind: F) -> T {
        may_unwind()
    }

    /// A test only type that will abort the program execution once dropped.
    ///
    /// To avoid aborting the proccess, callers must `forget` all instances of
    /// the `Abort` type.
    #[cfg(all(test, panic = "unwind"))]
    struct DebugAbort {
        location: &'static Location<'static>,
    }

    #[cfg(all(test, panic = "unwind"))]
    #[cfg(not(tarpaulin_include))]
    impl Drop for DebugAbort {
        fn drop(&mut self) {
            panic!("thread exits are forbidden inside {:?}, aborting", self.location);
        }
    }
}

pub mod hint {
    #[cfg(not(all(loom, test)))]
    pub use core::hint::spin_loop;

    #[cfg(all(loom, test))]
    pub use loom::hint::spin_loop;
}

pub mod thread {
    #[cfg(all(any(feature = "yield", test), not(loom)))]
    pub use std::thread::yield_now;

    #[cfg(all(loom, test))]
    pub use loom::thread::yield_now;

    #[cfg(not(all(loom, test)))]
    pub use std::thread::LocalKey;

    #[cfg(all(loom, test))]
    pub use loom::thread::LocalKey;
}
