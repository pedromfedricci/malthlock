pub mod models {
    use core::array;

    use loom::sync::Arc;
    use loom::{model, thread};

    use crate::test::{AsDeref, AsDerefMut, LockThen, TryLockThen};

    type Int = usize;
    // TODO: Three or more threads make lock models run for too long. It would
    // be nice to run a lock model with at least three threads because that
    // would cover a queue with head, tail and at least one more queue node
    // instead of a queue with just head and tail.
    const LOCKS: Int = 2;
    const TRY_LOCKS: Int = 3;

    /// Increments a shared integer.
    fn inc<L: LockThen<Target = Int>>(lock: &Arc<L>) {
        lock.lock_then(|mut data| *data.as_deref_mut() += 1);
    }

    /// Tries to increment a shared integer.
    fn try_inc<L: TryLockThen<Target = Int>>(lock: &Arc<L>) {
        lock.try_lock_then(|opt| opt.map(|mut data| *data.as_deref_mut() += 1));
    }

    /// Get the shared integer.
    fn get<L: LockThen<Target = Int>>(lock: &Arc<L>) -> Int {
        lock.lock_then(|data| *data.as_deref())
    }

    /// Evaluates that concurrent `try_lock` calls will serialize all mutations
    /// against the shared data, therefore no data races.
    pub fn try_lock_join<L: TryLockThen<Target = Int> + 'static>() {
        model(|| {
            const RUNS: Int = TRY_LOCKS;
            let lock = Arc::new(L::new(0));
            let handles: [_; RUNS] = array::from_fn(|_| {
                let lock = Arc::clone(&lock);
                thread::spawn(move || try_inc(&lock))
            });
            for handle in handles {
                handle.join().unwrap();
            }
            let value = get(&lock);
            assert!((1..=RUNS).contains(&value));
        });
    }

    /// Evaluates that concurrent `lock` calls will serialize all mutations
    /// against the shared data, therefore no data races.
    pub fn lock_join<L: LockThen<Target = Int> + 'static>() {
        model(|| {
            const RUNS: Int = LOCKS;
            let lock = Arc::new(L::new(0));
            let handles: [_; RUNS] = array::from_fn(|_| {
                let lock = Arc::clone(&lock);
                thread::spawn(move || inc(&lock))
            });
            for handle in handles {
                handle.join().unwrap();
            }
            let value = get(&lock);
            assert_eq!(RUNS, value);
        });
    }

    /// Evaluates that concurrent `lock` and `try_lock` calls will serialize
    /// all mutations against the shared data, therefore no data races.
    pub fn mixed_lock_join<L: TryLockThen<Target = Int> + 'static>() {
        model(|| {
            const RUNS: Int = LOCKS;
            let lock = Arc::new(L::new(0));
            let handles: [_; RUNS] = array::from_fn(|run| {
                let lock = Arc::clone(&lock);
                let f = if run % 2 == 0 { inc } else { try_inc };
                thread::spawn(move || f(&lock))
            });
            for handle in handles {
                handle.join().unwrap();
            }
            let value = get(&lock);
            assert!((1..=RUNS).contains(&value));
        });
    }
}
