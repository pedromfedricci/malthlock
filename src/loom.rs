pub mod models {
    use core::array;

    use loom::sync::Arc;
    use loom::{model, thread};

    use crate::test::{get, inc, try_inc, Int};
    use crate::test::{LockThen, TryLockThen};

    /// Get a copy of the shared integer, converting it to usize.
    ///
    /// Panics if the cast fails.
    fn get_unwrap<L>(lock: &Arc<L>) -> usize
    where
        L: LockThen<Target = Int>,
    {
        get(lock).try_into().unwrap()
    }

    // TODO: Three or more threads make lock models run for too long. It would
    // be nice to run a lock model with at least three threads because that
    // would cover a queue with head, tail and at least one more queue node
    // instead of a queue with just head and tail.
    const LOCKS: usize = 2;
    const TRY_LOCKS: usize = 3;

    /// Evaluates that concurrent `try_lock` calls will serialize all mutations
    /// against the shared data, therefore no data races.
    pub fn try_lock_join<L>()
    where
        L: TryLockThen<Target = Int> + 'static,
    {
        model(|| {
            const RUNS: usize = TRY_LOCKS;
            let lock = Arc::new(L::new(0));
            let handles: [_; RUNS] = array::from_fn(|_| {
                let lock = Arc::clone(&lock);
                thread::spawn(move || try_inc(&lock))
            });
            for handle in handles {
                handle.join().unwrap();
            }
            let value = get_unwrap(&lock);
            assert!((1..=RUNS).contains(&value));
        });
    }

    /// Evaluates that concurrent `lock` calls will serialize all mutations
    /// against the shared data, therefore no data races.
    pub fn lock_join<L>()
    where
        L: LockThen<Target = Int> + 'static,
    {
        model(|| {
            const RUNS: usize = LOCKS;
            let lock = Arc::new(L::new(0));
            let handles: [_; RUNS] = array::from_fn(|_| {
                let lock = Arc::clone(&lock);
                thread::spawn(move || inc(&lock))
            });
            for handle in handles {
                handle.join().unwrap();
            }
            let value = get_unwrap(&lock);
            assert_eq!(RUNS, value);
        });
    }

    /// Evaluates that concurrent `lock` and `try_lock` calls will serialize
    /// all mutations against the shared data, therefore no data races.
    pub fn mixed_lock_join<L>()
    where
        L: TryLockThen<Target = Int> + 'static,
    {
        model(|| {
            const RUNS: usize = LOCKS;
            let lock = Arc::new(L::new(0));
            let handles: [_; RUNS] = array::from_fn(|run| {
                let lock = Arc::clone(&lock);
                let f = if run % 2 == 0 { inc } else { try_inc };
                thread::spawn(move || f(&lock))
            });
            for handle in handles {
                handle.join().unwrap();
            }
            let value = get_unwrap(&lock);
            assert!((1..=RUNS).contains(&value));
        });
    }
}
