// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <gtest/gtest.h>
#include <string>
#include <thread>
#include <mutex>

#include <semaphore_wrapper.hpp>
#include <utils.hpp>

const static std::string TEST_SEM_NAME{ "TestSemaphore" };

TEST(Semaphore, PV)
{
    std::string sem_name = get_thread_unique_name(TEST_SEM_NAME);
    NamedSemaphore sem(NamedSemaphore::OpenType::e_create, sem_name);
    bool ret;
    ret = sem.post();
    ASSERT_TRUE(ret);
    ret = sem.wait();
    ASSERT_TRUE(ret);
}

/**
 * @brief Test Semaphore can be shared by name
 * The initial value is 0, if cannot shared by name, this test will be blocked.
 */
TEST(Semaphore, SharedByName)
{
    unsigned int sem_init_value = 0;
    std::string sem_name = get_thread_unique_name(TEST_SEM_NAME);
    NamedSemaphore sem_created(NamedSemaphore::OpenType::e_create, sem_name,
                               sem_init_value);
    NamedSemaphore sem_opened(NamedSemaphore::OpenType::e_open, sem_name,
                              sem_init_value);
    bool post_result = sem_created.post();
    ASSERT_TRUE(post_result);
    bool wait_result = sem_opened.wait();
    ASSERT_TRUE(wait_result);
}

TEST(Semaphore, Release)
{
    std::string sem_name = get_thread_unique_name(TEST_SEM_NAME);
    {
        NamedSemaphore sem_created(NamedSemaphore::OpenType::e_open_or_create,
                                   sem_name);
        // Semaphore should be destried here
    }
    auto open_nonexist_sem = [&]() {
        // The semaphore is allready destried, cannot be opened here
        NamedSemaphore sem_opened(NamedSemaphore::OpenType::e_open, sem_name);
    };
    ASSERT_THROW({ open_nonexist_sem(); }, std::runtime_error);
}

/**
 * P will be called first and then V
 */
TEST(Semaphore, PFirst)
{
    std::string sem_name = get_thread_unique_name(TEST_SEM_NAME);
    NamedSemaphore sem(NamedSemaphore::OpenType::e_create, sem_name);
    auto poster = [&]() {
        bool wait_result = sem.wait();
        ASSERT_TRUE(wait_result);
    };
    std::thread th(poster);
    bool post_result = sem.post();
    ASSERT_TRUE(post_result);
    th.join();
}

/**
 * Test the OpenType option
 */
TEST(Semaphore, CreateOrOpen)
{
    std::string sem_name = get_thread_unique_name(TEST_SEM_NAME);
    unsigned int sem_init_value = 0;
    NamedSemaphore sem_created(NamedSemaphore::OpenType::e_open_or_create,
                               sem_name, sem_init_value);
    NamedSemaphore sem_opened(NamedSemaphore::OpenType::e_open_or_create,
                              sem_name, sem_init_value);
    bool post_result = sem_created.post();
    ASSERT_TRUE(post_result);
    bool wait_result = sem_opened.wait();
    ASSERT_TRUE(wait_result);
}

TEST(Semaphore, TryWait)
{
    std::string sem_name = get_thread_unique_name(TEST_SEM_NAME);
    NamedSemaphore sem(NamedSemaphore::OpenType::e_open_or_create, sem_name);
    bool wait_result;
    wait_result = sem.try_wait();
    ASSERT_FALSE(wait_result);
    sem.post();
    wait_result = sem.try_wait();
    ASSERT_TRUE(wait_result);
}

TEST(Semaphore, Lock)
{
    std::string sem_name = get_thread_unique_name(TEST_SEM_NAME);
    SemaphoreMutex mutex(NamedSemaphore::OpenType::e_open_or_create, sem_name);
    {
        std::lock_guard<SemaphoreMutex> lock(mutex);
        bool lock_result = mutex.try_lock(); // Should fail because of lock
        ASSERT_FALSE(lock_result);
    }
    bool lock_result = mutex.try_lock();
    ASSERT_TRUE(
        lock_result); // Should success because lock will release the semaphore
}

/**
 * @brief Test open the same semaphore by name
 * 
 */
TEST(Semaphore, MutexOpen)
{
    std::string sem_name = get_thread_unique_name(TEST_SEM_NAME);
    SemaphoreMutex create_mutex(NamedSemaphore::OpenType::e_create, sem_name);
    SemaphoreMutex open_mutex(NamedSemaphore::OpenType::e_open, sem_name);
    open_mutex.lock();
    create_mutex.unlock();
    bool lock_result = open_mutex.try_lock();
    ASSERT_TRUE(lock_result);
}