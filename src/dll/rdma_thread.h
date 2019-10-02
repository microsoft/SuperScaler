#ifndef RDMA_THREAD_H_
#define RDMA_THREAD_H_

#include "rdma_common.h"

namespace wolong {

/// \brief Options to configure a Thread.
///
/// Note that the options are all hints, and the
/// underlying implementation may choose to ignore it.
struct ThreadOptions {
	/// Thread stack size to use (in bytes).
	size_t stack_size = 0;  // 0: use system default value
	/// Guard area size to use near thread stacks to use (in bytes)
	size_t guard_size = 0;  // 0: use system default value
};

class Thread {
 public:
	Thread() {}

	/// Blocks until the thread of control stops running.
	virtual ~Thread() {}
 private:
	WOLONG_DISALLOW_COPY_AND_ASSIGN(Thread);
};

extern 
Thread* StartThread(
	const ThreadOptions& thread_options, 
	const std::string& name,										
	std::function<void()> fn);

}

#endif // RDMA_THREAD_H_
