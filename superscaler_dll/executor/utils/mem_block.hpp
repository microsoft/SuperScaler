#pragma once

#include <cstddef>

/**
 * @brief A class to represent dynamic allocated memory block by base, offset and length.
 */
class MemBlock {
public:
	MemBlock() = delete;
	MemBlock(const void *address, size_t length)
		: m_base(address), m_offset(0), m_length(length) {}
	MemBlock(const void *base, size_t offset, size_t length)
		: m_base(base), m_offset(offset), m_length(length) {}
	~MemBlock() = default;

	/**
	 * @brief Get constant base address
	 */
	const void *get_base() const
	{
		return m_base;
	}

	/**
	 * @brief Get base address
	 */
	void *get_base()
	{
		return const_cast<void *>(m_base);
	}

	/**
	 * @brief Get offset from base address
	 */
	size_t get_offset() const
	{
		return m_offset;
	}

	size_t get_length() const
	{
		return m_length;
	}

	/**
	 * @brief Get constant actual address
	 */
	const void *get_address() const
	{
		return (const char *)m_base + m_offset;
	}

	/**
	 * @brief Get actual address
	 */
	void *get_address()
	{
		return (char *)m_base + m_offset;
	}

private:
	const void *m_base;
	size_t m_offset;
	size_t m_length;
};
