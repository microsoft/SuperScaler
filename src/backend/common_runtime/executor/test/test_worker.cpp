// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <thread>
#include <vector>
#include <random>
#include <unordered_map>

#include <gtest/gtest.h>

#include <task.hpp>
#include <worker.hpp>

TEST(Worker, AddLambdaTask)
{
    bool success = false;
    Worker w;
    w.add_task([&success](TaskState) { success = true; });
    w.exit();
    ASSERT_TRUE(success);
}

TEST(Worker, AddTaskObject)
{
    bool success = false;
    Worker w;
    auto t = std::make_shared<Task>(nullptr,
                                    [&success](TaskState) { success = true; });
    w.add_task(t);
    w.exit();
    ASSERT_TRUE(success);
}

static bool task_func_success = false;
static void task_func(TaskState)
{
    task_func_success = true;
}
TEST(Worker, AddFunctionPointer)
{
    Worker w;
    w.add_task(task_func);
    w.exit();
    ASSERT_TRUE(task_func_success);
}

TEST(Worker, RandomAddTask)
{
    Worker w;
    size_t counter = 0;
    size_t target = 256;
    std::vector<std::thread> threads;
    for (size_t i = 0; i < target; ++i) {
        threads.push_back(std::thread([&counter, &w] {
            w.add_task([&counter](TaskState) { counter++; });
        }));
    }
    for (auto &t : threads) {
        t.join();
    }
    w.exit();
    ASSERT_EQ(counter, target);
}

TEST(Worker, RecursiveAddTask)
{
    bool success = false;
    Worker w;
    auto t1 = std::make_shared<Task>(nullptr, [&success, &w](TaskState) {
        auto t2 = std::make_shared<Task>(nullptr, [&success](TaskState) {
            success = true;
        });
        w.add_task(t2);
    });
    w.add_task(t1);
    t1->wait();
    w.exit();
    ASSERT_TRUE(success);
}
