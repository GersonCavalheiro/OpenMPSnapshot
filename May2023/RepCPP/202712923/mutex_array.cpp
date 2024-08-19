

#include <iostream>
#include <thrust/reduce.h>
#include <thrust/sequence.h>

#include <stdgpu/atomic.cuh> 
#include <stdgpu/iterator.h> 
#include <stdgpu/memory.h>   
#include <stdgpu/mutex.cuh>  
#include <stdgpu/platform.h> 
#include <stdgpu/vector.cuh> 

struct is_odd
{
STDGPU_HOST_DEVICE bool
operator()(const int x) const
{
return x % 2 == 1;
}
};

void
try_partial_sum(const int* d_input, const stdgpu::index_t n, stdgpu::mutex_array<>& locks, int* d_result)
{
#pragma omp parallel for
for (stdgpu::index_t i = 0; i < n; ++i)
{
stdgpu::index_t j = i % locks.size();

bool finished = false;
const stdgpu::index_t number_trials = 5;
for (stdgpu::index_t k = 0; k < number_trials; ++k)
{
if (!finished && locks[j].try_lock())
{

d_result[j] += d_input[i];

locks[j].unlock();
finished = true;
}
}
}
}

int
main()
{

const stdgpu::index_t n = 100;
const stdgpu::index_t m = 10;

int* d_input = createDeviceArray<int>(n);
int* d_result = createDeviceArray<int>(m);
stdgpu::mutex_array<> locks = stdgpu::mutex_array<>::createDeviceObject(m);

thrust::sequence(stdgpu::device_begin(d_input), stdgpu::device_end(d_input), 1);


try_partial_sum(d_input, n, locks, d_result);

int sum = thrust::reduce(stdgpu::device_cbegin(d_result), stdgpu::device_cend(d_result), 0, thrust::plus<int>());

const int sum_closed_form = n * (n + 1) / 2;

std::cout << "The sum of all partially computed sums (via mutex locks) is " << sum
<< " which intentionally might not match the expected value of " << sum_closed_form << std::endl;

destroyDeviceArray<int>(d_input);
destroyDeviceArray<int>(d_result);
stdgpu::mutex_array<>::destroyDeviceObject(locks);
}
