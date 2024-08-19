
























#pragma once
#include <stdio.h>

#include <cstdint>

class PoolAllocator {
public:
PoolAllocator() {}
~PoolAllocator() {}
void init(sycl::queue &q) try {
memoryUtil::deviceAlloc(q, d_pool, MAX_SIZE);
memoryUtil::deviceSet(q, d_pool, MAX_SIZE, 0x00);
memoryUtil::deviceAlloc(q, d_count, 1);
memoryUtil::deviceSet(q, d_count, uint32_t(1), 0x00);
}
catch (sycl::exception const &exc) {
std::cerr << exc.what() << "\nException caught at file:" << __FILE__
<< ", line:" << __LINE__ << std::endl;
}

void free(sycl::queue &q) try {
memoryUtil::deviceFree(q, d_pool);
memoryUtil::deviceFree(q, d_count);
}
catch (sycl::exception const &exc) {
std::cerr << exc.what() << "\nException caught at file:" << __FILE__
<< ", line:" << __LINE__ << std::endl;
}

double compute_usage(sycl::queue &q) try {
uint32_t allocations_count;
memoryUtil::cpyToHost(q, d_count, &allocations_count, 1);
double num_bytes = double(allocations_count) * NODE_SIZE * sizeof(uint32_t);
return num_bytes / (1u << 30);
}
catch (sycl::exception const &exc) {
std::cerr << exc.what() << "\nException caught at file:" << __FILE__
<< ", line:" << __LINE__ << std::endl;
return 0;
}

PoolAllocator& operator=(const PoolAllocator& rhs) {
d_pool = rhs.d_pool;
d_count = rhs.d_count;
return *this;
}

template <typename AddressT = uint32_t>
inline void freeAddress(AddressT &address) {}

uint32_t getCapacity() { return MAX_SIZE; }

uint32_t getOffset() { return *d_count; }

uint32_t* getPool() { return d_pool; }

uint32_t* getCount() { return d_count; }

private:
uint32_t* d_pool;
uint32_t* d_count;

static constexpr uint64_t NODE_SIZE = 32;
static constexpr uint64_t MAX_NODES = 1 << 25;
static constexpr uint64_t MAX_SIZE = MAX_NODES * NODE_SIZE;  
};

