#include <functional>
#include <memory>
#include <gtest/gtest.h>

#include <task.hpp>

class MockTask : public Task {
public:
    MockTask(bool &success, std::function<void(void)> callback)
        : Task(nullptr, callback), m_success(success)
    {
    }

private:
    bool &m_success;

    void execute(Executor *)
    {
        m_success = true;
    }
};

TEST(Task, ExecuteAndCallback)
{
    bool execute_success = false, callback_success = false;
    MockTask task(execute_success,
                  [&callback_success] { callback_success = true; });
    task();
    task.wait();
    ASSERT_TRUE(execute_success);
    ASSERT_TRUE(callback_success);
}
