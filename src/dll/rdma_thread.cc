#include "rdma_thread.h"

namespace wolong {

class StdThread : public Thread {
 public:
	// name and thread_options are both ignored.
	StdThread(const ThreadOptions& thread_options, const std::string& name,
						std::function<void()> fn)
		: thread_(fn) {}
	~StdThread() { thread_.join(); }

 private:
	std::thread thread_;
};

Thread* StartThread(
	const ThreadOptions& thread_options, 
	const std::string& name,										
	std::function<void()> fn) 
{
	return new StdThread(thread_options, name, fn);
}

}
