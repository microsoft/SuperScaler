#include <array>
#include <random>
#include <gtest/gtest.h>

#include "../utils/mem_block.hpp"

constexpr int test_size = 128;

TEST(MemBlock, GeneralTest)
{
	std::array<char, test_size> test_data;
	for (int i = 0; i < test_size; i++)
		test_data[i] = (i % 128);
	size_t length = 10;
	
	MemBlock start_addr(test_data.data(), length);
	size_t offset = std::rand() % test_size;
	MemBlock middle_addr(test_data.data(), offset, length);

	char *start = (char *)start_addr.get_address();
	char *middle = (char *)middle_addr.get_address();

	ASSERT_EQ(*start, 0);
	ASSERT_EQ(*middle, offset);
	ASSERT_EQ(middle_addr.get_offset(), offset);
	ASSERT_EQ(start_addr.get_length(), length);
	ASSERT_EQ(middle_addr.get_length(), length);

	start = (char *)middle_addr.get_base();
	ASSERT_EQ(*start, 0);
}
