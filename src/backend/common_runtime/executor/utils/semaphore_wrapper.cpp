// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <stdexcept>
#include <errno.h>
#include <cstdio>
#include <cstring>

#include "semaphore_wrapper.hpp"

const mode_t SEMAPHORE_MODE = 0644;

NamedSemaphore::NamedSemaphore(NamedSemaphore::OpenType open_type,
                               const std::string &name, unsigned int init_size)
    : m_name(name), m_own(false)
{
    int open_flag = 0;
    int create_flag = O_CREAT | O_EXCL;
    switch (open_type) {
    case OpenType::e_open_or_create:
        //Try to create first
        m_handle =
            sem_open(m_name.c_str(), create_flag, SEMAPHORE_MODE, init_size);
        if (m_handle != SEM_FAILED) {
            // If create succeed
            m_own = true;
            break;
        }
        if (errno != EEXIST) {
            // Not because the semaphore is already exist
            break;
        }
        // Try to Open it
    case OpenType::e_open:
        m_handle = sem_open(m_name.c_str(), open_flag);
        break;
    case OpenType::e_create:
        m_own = true;
        m_handle =
            sem_open(m_name.c_str(), create_flag, SEMAPHORE_MODE, init_size);
        break;
    default:
        //Should never get here
        throw std::runtime_error(
            std::string() + "Init NamedSemaphore failed, unknown opentype");
    }
    if (m_handle == SEM_FAILED) {
        throw std::runtime_error(
            std::string() + "Init NamedSemaphore failed: " + strerror(errno));
    }
}

NamedSemaphore::~NamedSemaphore()
{
    if (m_handle == SEM_FAILED)
        return;
    // The semaphore owner should release the resource
    if (m_own) {
        // The semaphore will be destoried after all process closed
        remove(m_name);
    }
    sem_close(m_handle);
}

bool NamedSemaphore::wait()
{
    int ret = sem_wait(m_handle);
    if (ret == 0)
        return true;
    if (errno == EINVAL) {
        // Not a valid semaphore
        throw std::runtime_error(std::string() + "sem wait error: " +
                                 strerror(errno) + ". At:" + __func__);
    }
    return false;
}

bool NamedSemaphore::post()
{
    int ret = sem_post(m_handle);
    if (ret == 0)
        return true;
    if (errno == EINVAL) {
        // Not a valid semaphore
        throw std::runtime_error(std::string() + "sem post error: " +
                                 strerror(errno) + ". At:" + __func__);
    }
    return false;
}

bool NamedSemaphore::try_wait()
{
    int ret = sem_trywait(m_handle);
    if (ret == 0)
        return true;
    if (errno != EAGAIN) {
        throw std::runtime_error(std::string() + "Try wait failed: " +
                                 strerror(errno) + ". At:" + __func__);
    }
    return false;
}

bool NamedSemaphore::remove(const std::string &name) noexcept
{
    int ret = sem_unlink(name.c_str());
    if (ret == 0)
        return true;
    return false;
}

std::string NamedSemaphore::get_name() const
{
    return m_name;
}

SemaphoreMutex::SemaphoreMutex(NamedSemaphore::OpenType open_type,
                               const std::string &name)
    : NamedSemaphore(open_type, name, 1)
{
}

void SemaphoreMutex::lock()
{
    bool ret = wait();
    if (!ret) {
        throw std::runtime_error(
            std::string() +
            "Semaphore lock failed, reason: " + strerror(errno));
    }
}

void SemaphoreMutex::unlock()
{
    bool ret = post();
    if (!ret) {
        throw std::runtime_error(
            std::string() +
            "Semaphore unlock failed, reason: " + strerror(errno));
    }
}

bool SemaphoreMutex::try_lock()
{
    return try_wait();
}