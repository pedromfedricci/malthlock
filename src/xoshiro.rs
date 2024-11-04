use rand_xoshiro::rand_core::{RngCore, SeedableRng};
use rand_xoshiro::{Seed512, Xoshiro512StarStar};

use crate::cfg::cell::{UnsafeCell, UnsafeCellWith};
use crate::cfg::thread::LocalKey;
use crate::fairness::Fairness;

/// A random generator.
struct Generator {
    inner: UnsafeCell<Xoshiro512StarStar>,
}

impl Generator {
    /// Creates a new, seeded `Generator` instance.
    fn new() -> Self {
        let seed = Seed512::default();
        let gen = Xoshiro512StarStar::from_seed(seed);
        let inner = UnsafeCell::new(gen);
        Self { inner }
    }
}

/// A thread local random generator.
pub struct LocalGenerator {
    #[cfg(not(all(loom, test)))]
    key: LocalKey<Generator>,

    // We can't take ownership of Loom's `thread_local!` value since it is a
    // `static`, non-copy value, so we just point to it.
    #[cfg(all(loom, test))]
    key: &'static LocalKey<Generator>,
}

impl LocalGenerator {
    /// Returns next random `u64` from thread local generator.
    fn next_u64(&'static self) -> u64 {
        let next_u64 = |gen: &mut Xoshiro512StarStar| gen.next_u64();
        // SAFETY: This key is only ever accessed within its own thread, and it
        // is not aliased when the cast to `&mut T` (by UnsafeCell) happens.
        self.key.with(|gen| unsafe { gen.inner.with_mut_unchecked(next_u64) })
    }
}

#[cfg(not(all(loom, test)))]
const GENERATOR: LocalGenerator = {
    std::thread_local! {
        static GENERATOR: Generator = Generator::new();
    }
    LocalGenerator { key: GENERATOR }
};

#[cfg(all(loom, test))]
static GENERATOR: LocalGenerator = {
    loom::thread_local! {
        static GENERATOR: Generator = Generator::new();
    }
    LocalGenerator { key: &GENERATOR }
};

impl Fairness for LocalGenerator {
    fn get() -> &'static Self {
        &GENERATOR
    }

    #[cfg(not(test))]
    fn should_promote_passive_tail(this: &'static Self) -> bool {
        this.next_u64() == 0
    }

    #[cfg(test)]
    fn should_promote_passive_tail(this: &'static Self) -> bool {
        // For testing purposes, lets make this happen often.
        this.next_u64() % 2 == 0
    }
}
