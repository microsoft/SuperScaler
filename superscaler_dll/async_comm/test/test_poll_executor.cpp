#include <memory>
#include <gtest/gtest.h>

#include <task.hpp>
#include <worker.hpp>
#include <poll_executor.hpp>


TEST(PollExecutor, AddTask) {
    bool success = false;
    PollExecutor exec;
    auto t = std::make_shared<Task>(&exec, [&success] { success = true; });
    exec.add_task(t);
    t->wait();
    ASSERT_TRUE(success);
}

TEST(PollExecutor, RecursiveAddTask) {
    bool success = false;
    PollExecutor exec;
    auto t = std::make_shared<Task>(&exec, [&exec, &success] {
        auto it = std::make_shared<Task>(&exec, [&success]{
            success = true;
        });
        exec.add_task(it, true);
        it->wait();
    });
    exec.add_task(t, true);
    t->wait();
    ASSERT_TRUE(success);
}
