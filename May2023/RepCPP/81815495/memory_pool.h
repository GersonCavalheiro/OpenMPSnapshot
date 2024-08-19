

#if !defined(KRATOS_MEMORY_POOL_H_INCLUDED )
#define  KRATOS_MEMORY_POOL_H_INCLUDED

#include <memory>
#include <vector>
#include <sstream>

#include "includes/fixed_size_memory_pool.h"

namespace Kratos
{



class MemoryPool
{
public:    

using PoolsContainerType = std::vector<FixedSizeMemoryPool*>;


MemoryPool(MemoryPool const& rOther) = delete;

virtual ~MemoryPool() {
for (auto i_pool = GetInstance().mPools.begin(); i_pool != GetInstance().mPools.end(); i_pool++)
delete *i_pool;
}


MemoryPool& operator=(MemoryPool const& rOther) = delete;



static void* Allocate(std::size_t ObjectSizeInBytes) {
return GetPoolWithBlockSize(ObjectSizeInBytes)->Allocate();
}

static void Deallocate(void* pPointrerToRelease, std::size_t ObjectSizeInBytes) {
GetPoolWithBlockSize(ObjectSizeInBytes)->Deallocate(pPointrerToRelease);
}



static MemoryPool& GetInstance() {
static MemoryPool instance;
return instance;
}

static std::size_t GetNumberOfPools()  {
return GetInstance().mPools.size();
}

static FixedSizeMemoryPool* GetPoolWithBlockSize(std::size_t BlockSize) {
PoolsContainerType& r_pools = GetInstance().mPools;

if (r_pools.size() <= BlockSize) { 
#pragma omp critical
{
if (r_pools.size() <= BlockSize) 
r_pools.resize(BlockSize + 1, nullptr);
}
}

if (r_pools[BlockSize] == nullptr) { 
#pragma omp critical
{
if (r_pools[BlockSize] == nullptr) 
r_pools[BlockSize] = new FixedSizeMemoryPool(BlockSize);
}
}
return r_pools[BlockSize];
}



static std::size_t MemoryUsed() {
std::size_t result = sizeof(MemoryPool);
for (auto i_pool = GetInstance().mPools.begin(); i_pool != GetInstance().mPools.end(); i_pool++)
if(*i_pool != nullptr)
result += (*i_pool)->MemoryUsed();
return result;
}

static std::size_t MemoryOverhead() {
std::size_t result = sizeof(MemoryPool);
for (auto i_pool = GetInstance().mPools.begin(); i_pool != GetInstance().mPools.end(); i_pool++)
if (*i_pool != nullptr)
result += (*i_pool)->MemoryOverhead();
return result;
}


static std::string Info()  {
std::stringstream buffer("MemoryPool");
std::size_t memory_used = MemoryUsed();
std::size_t memory_overhead = MemoryOverhead();
double overhead_percentage = memory_overhead;
if (memory_overhead < memory_used)
overhead_percentage = static_cast<double>(memory_overhead)/(memory_used - memory_overhead);
overhead_percentage *= 100.00;

buffer << "Total memory usage: " 
<< SizeInBytesToString(MemoryUsed()) << " bytes and memory overhead " 
<< SizeInBytesToString(MemoryOverhead()) << "(" << overhead_percentage << "%)" << std::endl;

return buffer.str();
}




private:


MemoryPool(){}


PoolsContainerType mPools;


static std::string SizeInBytesToString(std::size_t Bytes) {
std::stringstream buffer;
double result = Bytes;
constexpr int units_size = 5;
constexpr char units[units_size+1] = { ' ', 'k','M','G','T', 'T'};
int i = 0;
for (; i < units_size; i++)
if (result > 1024)
{
result /= 1024;
}
else
break;
buffer << result << units[i];

return buffer.str();
}


}; 




}  

#endif 
