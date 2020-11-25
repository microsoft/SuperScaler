// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <functional>
#include <memory>
#include <gtest/gtest.h>

#include <task.hpp>

class MockTask : public Task {
public:
    MockTask(bool &success, std::function<void(TaskState)> callback)
        : Task(nullptr, callback), m_success(success)
    {
    }

private:
    bool &m_success;

    TaskState execute(Executor *)
    {
        m_success = true;
        return TaskState::e_success;
    }
};

TEST(Task, ExecuteAndCallback)
{
    bool execute_success = false, callback_success = false;
    MockTask task(execute_success,
                  [&callback_success](TaskState) { callback_success = true; });
    task();
    task.wait();
    ASSERT_TRUE(execute_success);
    ASSERT_TRUE(callback_success);
}

TEST(Task, WaitBeforeCommit)
{
    bool execute_success = false, callback_success = false;
    MockTask task(execute_success,
                  [&callback_success](TaskState) { callback_success = true; });
    TaskState state;
    // Wait before commit
    state = task.wait();
    ASSERT_EQ(state, TaskState::e_uncommitted);
    task.commit();
    task();
    state = task.wait();
    ASSERT_TRUE(execute_success);
    ASSERT_TRUE(callback_success);
    ASSERT_EQ(state, TaskState::e_success);
}
