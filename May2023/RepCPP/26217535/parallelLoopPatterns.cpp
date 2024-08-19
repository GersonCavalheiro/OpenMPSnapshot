

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>

#include <iostream>
#include <typeinfo>

ALPAKA_FN_HOST_ACC float process(uint32_t idx)
{
return static_cast<float>(idx + 3);
}

template<typename TQueue, typename TBufAcc>
void testResult(TQueue& queue, TBufAcc& bufAcc)
{
alpaka::wait(queue);
auto const n = alpaka::getExtentProduct(bufAcc);
auto const devHost = alpaka::getDevByIdx<alpaka::DevCpu>(0u);
auto bufHost = alpaka::allocBuf<float, uint32_t>(devHost, n);
alpaka::memcpy(queue, bufHost, bufAcc);
auto const byte(static_cast<uint8_t>(0u));
alpaka::memset(queue, bufAcc, byte);
auto const* result = alpaka::getPtrNative(bufHost);
bool testPassed = true;
for(uint32_t i = 0u; i < n; i++)
testPassed = testPassed && (std::abs(result[i] - process(i)) < 1e-3);
std::cout << (testPassed ? "Test passed.\n" : "Test failed.\n");
}

using WorkDiv = alpaka::WorkDivMembers<alpaka::DimInt<1u>, uint32_t>;

struct NaiveCudaStyleKernel
{
template<typename TAcc>
ALPAKA_FN_ACC void operator()(TAcc const& acc, float* result, uint32_t n) const
{
auto const globalThreadIdx(alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u]);
if(globalThreadIdx < n)
{
auto const dataDomainIdx = globalThreadIdx;
auto const memoryIdx = globalThreadIdx;
result[memoryIdx] = process(dataDomainIdx);
}
}
};

template<typename TAcc, typename TQueue, typename TBufAcc>
void naiveCudaStyle(TQueue& queue, TBufAcc& bufAcc)
{
auto const n = alpaka::getExtentProduct(bufAcc);
auto const deviceProperties = alpaka::getAccDevProps<TAcc>(alpaka::getDevByIdx<TAcc>(0u));
auto const maxThreadsPerBlock = deviceProperties.m_blockThreadExtentMax[0];

auto const threadsPerBlock = maxThreadsPerBlock;
auto const blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
auto const elementsPerThread = 1u;
auto workDiv = WorkDiv{blocksPerGrid, threadsPerBlock, elementsPerThread};
std::cout << "\nNaive CUDA style processing - each thread processes one data point:\n";
std::cout << "   " << blocksPerGrid << " blocks, " << threadsPerBlock << " threads per block, "
<< "alpaka element layer not used\n";
alpaka::exec<TAcc>(queue, workDiv, NaiveCudaStyleKernel{}, alpaka::getPtrNative(bufAcc), n);
testResult(queue, bufAcc);
}

struct GridStridedLoopKernel
{
template<typename TAcc>
ALPAKA_FN_ACC void operator()(TAcc const& acc, float* result, uint32_t n) const
{
auto const globalThreadExtent(alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0u]);
auto const globalThreadIdx(alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u]);
for(uint32_t dataDomainIdx = globalThreadIdx; dataDomainIdx < n; dataDomainIdx += globalThreadExtent)
{
auto const memoryIdx = dataDomainIdx;
result[memoryIdx] = process(dataDomainIdx);
}
}
};

template<typename TAcc, typename TQueue, typename TBufAcc>
void gridStridedLoop(TQueue& queue, TBufAcc& bufAcc)
{
auto const n = alpaka::getExtentProduct(bufAcc);
auto const deviceProperties = alpaka::getAccDevProps<TAcc>(alpaka::getDevByIdx<TAcc>(0u));
auto const maxThreadsPerBlock = deviceProperties.m_blockThreadExtentMax[0];

auto const threadsPerBlock = maxThreadsPerBlock;
auto const blocksPerGrid = deviceProperties.m_multiProcessorCount;
auto const elementsPerThread = 1u;
auto workDiv = WorkDiv{blocksPerGrid, threadsPerBlock, elementsPerThread};
std::cout << "\nGrid strided loop processing - fixed number of threads and blocks:\n";
std::cout << "   " << blocksPerGrid << " blocks, " << threadsPerBlock << " threads per block, "
<< "alpaka element layer not used\n";
alpaka::exec<TAcc>(queue, workDiv, GridStridedLoopKernel{}, alpaka::getPtrNative(bufAcc), n);
testResult(queue, bufAcc);
}

struct ChunkedGridStridedLoopKernel
{
template<typename TAcc>
ALPAKA_FN_ACC void operator()(TAcc const& acc, float* result, uint32_t n) const
{
auto const numElements(alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[0u]);
auto const globalThreadExtent(alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0u]);
auto const globalThreadIdx(alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u]);
for(uint32_t chunkStart = globalThreadIdx * numElements; chunkStart < n;
chunkStart += globalThreadExtent * numElements)
{
for(uint32_t dataDomainIdx = chunkStart; (dataDomainIdx < chunkStart + numElements) && (dataDomainIdx < n);
dataDomainIdx++)
{
auto const memoryIdx = dataDomainIdx;
result[memoryIdx] = process(dataDomainIdx);
}
}
}
};

template<typename TAcc, typename TQueue, typename TBufAcc>
void chunkedGridStridedLoop(TQueue& queue, TBufAcc& bufAcc)
{
auto const n = alpaka::getExtentProduct(bufAcc);
auto const deviceProperties = alpaka::getAccDevProps<TAcc>(alpaka::getDevByIdx<TAcc>(0u));
auto const maxThreadsPerBlock = deviceProperties.m_blockThreadExtentMax[0];

auto const threadsPerBlock = maxThreadsPerBlock;
auto const blocksPerGrid = deviceProperties.m_multiProcessorCount;
auto const elementsPerThread = 8u;
auto workDiv = WorkDiv{blocksPerGrid, threadsPerBlock, elementsPerThread};
std::cout << "\nChunked grid strided loop processing - fixed number of threads and blocks:\n";
std::cout << "   " << blocksPerGrid << " blocks, " << threadsPerBlock << " threads per block, "
<< elementsPerThread << " alpaka elements per thread\n";
alpaka::exec<TAcc>(queue, workDiv, ChunkedGridStridedLoopKernel{}, alpaka::getPtrNative(bufAcc), n);
testResult(queue, bufAcc);
}

struct NaiveOpenMPStyleKernel
{
template<typename TAcc>
ALPAKA_FN_ACC void operator()(TAcc const& acc, float* result, uint32_t n) const
{
auto const globalThreadExtent(alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0u]);
auto const globalThreadIdx(alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u]);
auto const processPerThread = (n + globalThreadExtent - 1) / globalThreadExtent;
for(uint32_t dataDomainIdx = globalThreadIdx * processPerThread;
(dataDomainIdx < (globalThreadIdx + 1) * processPerThread) && (dataDomainIdx < n);
dataDomainIdx++)
{
auto const memoryIdx = dataDomainIdx;
result[memoryIdx] = process(dataDomainIdx);
}
}
};

template<typename TAcc, typename TQueue, typename TBufAcc>
void naiveOpenMPStyle(TQueue& queue, TBufAcc& bufAcc)
{
auto const n = alpaka::getExtentProduct(bufAcc);
auto const deviceProperties = alpaka::getAccDevProps<TAcc>(alpaka::getDevByIdx<TAcc>(0u));
auto const maxThreadsPerBlock = deviceProperties.m_blockThreadExtentMax[0];
auto const numCores = std::max(std::thread::hardware_concurrency(), 1u);

auto const threadsPerBlock = maxThreadsPerBlock;
auto const blocksPerGrid = numCores;
auto const elementsPerThread = 1u;
auto workDiv = WorkDiv{blocksPerGrid, threadsPerBlock, elementsPerThread};
std::cout << "\nNaive OpenMP style processing - each thread processes a single consecutive range of elements:\n";
std::cout << "   " << blocksPerGrid << " blocks, " << threadsPerBlock << " threads per block, "
<< "alpaka element layer not used\n";
alpaka::exec<TAcc>(queue, workDiv, NaiveOpenMPStyleKernel{}, alpaka::getPtrNative(bufAcc), n);
testResult(queue, bufAcc);
}

struct OpenMPSimdStyleKernel
{
template<typename TAcc>
ALPAKA_FN_ACC void operator()(TAcc const& acc, float* result, uint32_t n) const
{
auto const numElements(alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[0u]);
auto const globalThreadExtent(alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0u]);
auto const globalThreadIdx(alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u]);
auto const naiveProcessPerThread = (n + globalThreadExtent - 1) / globalThreadExtent;
auto const processPerThread = numElements * ((naiveProcessPerThread + numElements - 1) / numElements);
for(uint32_t chunkStart = globalThreadIdx * processPerThread;
chunkStart < (globalThreadIdx + 1) * processPerThread && (chunkStart < n);
chunkStart += numElements)
{
for(uint32_t dataDomainIdx = chunkStart; (dataDomainIdx < chunkStart + numElements) && (dataDomainIdx < n);
dataDomainIdx++)
{
auto const memoryIdx = dataDomainIdx;
result[memoryIdx] = process(dataDomainIdx);
}
}
}
};

template<typename TAcc, typename TQueue, typename TBufAcc>
void openMPSimdStyle(TQueue& queue, TBufAcc& bufAcc)
{
auto const n = alpaka::getExtentProduct(bufAcc);
auto const deviceProperties = alpaka::getAccDevProps<TAcc>(alpaka::getDevByIdx<TAcc>(0u));
auto const maxThreadsPerBlock = deviceProperties.m_blockThreadExtentMax[0];
auto const numCores = 16u; 

auto const threadsPerBlock = maxThreadsPerBlock;
auto const blocksPerGrid = numCores;
auto const elementsPerThread = 4u;
auto workDiv = WorkDiv{blocksPerGrid, threadsPerBlock, elementsPerThread};
std::cout << "\nOpenMP SIMD style processing - each thread processes a single consecutive range of elements:\n";
std::cout << "   " << blocksPerGrid << " blocks, " << threadsPerBlock << " threads per block, "
<< elementsPerThread << " alpaka elements per thread\n";
alpaka::exec<TAcc>(queue, workDiv, OpenMPSimdStyleKernel{}, alpaka::getPtrNative(bufAcc), n);
testResult(queue, bufAcc);
}

auto main() -> int
{
#if defined(ALPAKA_CI) && !defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
return EXIT_SUCCESS;
#else

using Dim = alpaka::DimInt<1u>;

using Acc = alpaka::ExampleDefaultAcc<Dim, uint32_t>;
std::cout << "Using alpaka accelerator: " << alpaka::getAccName<Acc>() << std::endl;

auto const devAcc = alpaka::getDevByIdx<Acc>(0u);
auto queue = alpaka::Queue<Acc, alpaka::Blocking>(devAcc);

uint32_t const bufferSize = 1317u;
auto bufAcc = alpaka::allocBuf<float, uint32_t>(devAcc, bufferSize);

naiveCudaStyle<Acc>(queue, bufAcc);
gridStridedLoop<Acc>(queue, bufAcc);
chunkedGridStridedLoop<Acc>(queue, bufAcc);
naiveOpenMPStyle<Acc>(queue, bufAcc);
openMPSimdStyle<Acc>(queue, bufAcc);

#endif
}
