#ifndef WOLONG_SPINLOCK_H_
#define WOLONG_SPINLOCK_H_

#include <cstdint>
#include <atomic>

#define SPINLOCK_UNLOCKED ((uint64_t)0)
#define SPINLOCK_LOCKED   ((uint64_t)1)

#define SPINLOCK_CONSTRUCTOR_INIT(var) var(SPINLOCK_UNLOCKED)

#define VERLOCK_LOCKBIT_MASK ((uint64_t)1)
#define VERLOCK_UNLOCKED_LSB ((uint64_t)0)
#define VERLOCK_LOCKED_LSB   ((uint64_t)1)
#define VERLOCK_ZERO         ((uint64_t)0)

#define VERLOCK_CONSTRUCTOR_INIT(var) var(VERLOCK_ZERO)

namespace wolong {

// a word-sized type
typedef std::atomic<uint64_t> spinlock_t;


inline void spinlock_init(spinlock_t *slock) {
	slock->store(SPINLOCK_UNLOCKED, std::memory_order_relaxed);
}

inline bool spinlock_try_acquire(spinlock_t *slock) {
	if(slock->load(std::memory_order_relaxed) == SPINLOCK_LOCKED) {
		return false;
	}

	uint64_t expected = SPINLOCK_UNLOCKED;

	return slock->compare_exchange_strong(
		expected, SPINLOCK_LOCKED, std::memory_order_acquire);
}

inline void spinlock_acquire(spinlock_t *slock) {
	while(!spinlock_try_acquire(slock)) {
		// spin here
		// TODO: we can be smarter and back-off for a while
		//       or check if there are any threads to switch to
	}
}

inline void spinlock_release(spinlock_t *slock) {
	slock->store(SPINLOCK_UNLOCKED, std::memory_order_release);
}


// a word-sized type
typedef std::atomic<uint64_t> verlock_t;


inline void verlock_init(verlock_t *vlock) {
	vlock->store(VERLOCK_ZERO, std::memory_order_relaxed);
}

inline bool verlock_is_locked(uint64_t val) {
	return (val & VERLOCK_LOCKED_LSB) != 0; 
}

inline bool verlock_lock_successful(uint64_t val) {
	return !verlock_is_locked(val);
}

// Try to acquire the lock. Returns the old value of the lock.
// verlock_lock_successful(ret) can be used to check if locking succeeded.
inline uint64_t verlock_try_acquire(verlock_t *vlock) {
	uint64_t val = vlock->load(std::memory_order_relaxed);

	if(verlock_is_locked(val)) {
		return val;
	}

	vlock->compare_exchange_strong(val, val + 1, std::memory_order_acquire);

	return val;
}

// Spins until locking succeeds. Returns old value of the lock.
inline uint64_t verlock_acquire(verlock_t *vlock) {
	uint64_t val;

	do {
		val = verlock_try_acquire(vlock);
	} while(verlock_is_locked(val));

	return val;
}

// Pass in the old version of the lock.
inline void verlock_release(verlock_t *vlock, uint64_t old_val) {
	vlock->store(old_val + 2, std::memory_order_release);
}

inline void verlock_release(verlock_t *vlock) {
	uint64_t val = vlock->load(std::memory_order_relaxed);
	vlock->store(val + 1, std::memory_order_release);
}

inline uint64_t verlock_read(verlock_t *vlock) {
	return vlock->load(std::memory_order_acquire);
}

}

#endif // WOLONG_SPINLOCK_H_
