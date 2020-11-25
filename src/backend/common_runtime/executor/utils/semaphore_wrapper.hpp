// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include <string>

#include <semaphore.h>
#include <fcntl.h>

class NamedSemaphore {
public:
    enum class OpenType { e_create, e_open, e_open_or_create };
    NamedSemaphore(OpenType open_type, const std::string &name,
                   unsigned int init_value = 0);
    virtual ~NamedSemaphore();

    /**
     * @brief Wait for semaphore
     * Throw exception when semaphore crash
     * @return true Wait success
     * @return false Failed, the errno is set indicate the error
     */
    bool wait();

    /**
     * @brief Post the semaphore
     * Throw exception when semaphore
     * @return true success
     * @return false Failed, the errno is set indicate the error
     */
    bool post();

    /**
     * @brief Try wait
     * 
     * @return true success
     * @return false cannot wait or semaphore error.
     */
    bool try_wait();

    std::string get_name() const;

    /**
     * @brief Use sem_unlink to remove the semaphore named \p name
     * As this function may be used in destructors, this function won't throw exceptions
     * @param name 
     * @return true when success, false when failed. Reason saved in errno
     */
    static bool remove(const std::string &name) noexcept;

private:
    std::string m_name;
    sem_t *m_handle;
    bool m_own;
};

class SemaphoreMutex : private NamedSemaphore {
public:
    SemaphoreMutex(NamedSemaphore::OpenType open_type, const std::string &name);
    void lock();
    void unlock();
    bool try_lock();
    std::string get_name() const;
};