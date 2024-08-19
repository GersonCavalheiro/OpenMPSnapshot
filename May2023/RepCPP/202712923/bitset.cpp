

#include <iostream>
#include <thrust/copy.h>
#include <thrust/sequence.h>

#include <stdgpu/atomic.cuh> 
#include <stdgpu/bitset.cuh> 
#include <stdgpu/iterator.h> 
#include <stdgpu/memory.h>   
#include <stdgpu/platform.h> 

struct is_odd
{
STDGPU_HOST_DEVICE bool
operator()(const int x) const
{
return x % 2 == 1;
}
};

void
set_bits(const int* d_result, const stdgpu::index_t d_result_size, stdgpu::bitset<>& bits, stdgpu::atomic<int>& counter)
{
#pragma omp parallel for
for (stdgpu::index_t i = 0; i < d_result_size; ++i)
{
bool was_set = bits.set(d_result[i]);

if (!was_set)
{
++counter;
}
}
}

int
main()
{

const stdgpu::index_t n = 100;

int* d_input = createDeviceArray<int>(n);
int* d_result = createDeviceArray<int>(n / 2);
stdgpu::bitset<> bits = stdgpu::bitset<>::createDeviceObject(n);
stdgpu::atomic<int> counter = stdgpu::atomic<int>::createDeviceObject();

thrust::sequence(stdgpu::device_begin(d_input), stdgpu::device_end(d_input), 1);


thrust::copy_if(stdgpu::device_cbegin(d_input),
stdgpu::device_cend(d_input),
stdgpu::device_begin(d_result),
is_odd());



counter.store(0);

set_bits(d_result, n / 2, bits, counter);


std::cout << "First run: The number of set bits is " << bits.count() << " (" << n / 2 << " expected; "
<< counter.load() << " of those previously unset)" << std::endl;

counter.store(0);

set_bits(d_result, n / 2, bits, counter);


std::cout << "Second run: The number of set bits is " << bits.count() << " (" << n / 2 << " expected; "
<< counter.load() << " of those previously unset)" << std::endl;

destroyDeviceArray<int>(d_input);
destroyDeviceArray<int>(d_result);
stdgpu::bitset<>::destroyDeviceObject(bits);
stdgpu::atomic<int>::destroyDeviceObject(counter);
}
