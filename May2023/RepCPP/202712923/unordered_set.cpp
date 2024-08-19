

#include <iostream>
#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>

#include <stdgpu/iterator.h>        
#include <stdgpu/memory.h>          
#include <stdgpu/platform.h>        
#include <stdgpu/unordered_set.cuh> 

struct is_odd
{
STDGPU_HOST_DEVICE bool
operator()(const int x) const
{
return x % 2 == 1;
}
};

void
insert_neighbors(const int* d_result, const stdgpu::index_t n, stdgpu::unordered_set<int>& set)
{
#pragma omp parallel for
for (stdgpu::index_t i = 0; i < n; ++i)
{
int num = d_result[i];
int num_neighborhood[3] = { num - 1, num, num + 1 };

for (int num_neighbor : num_neighborhood)
{
set.insert(num_neighbor);
}
}
}

int
main()
{

const stdgpu::index_t n = 100;

int* d_input = createDeviceArray<int>(n);
int* d_result = createDeviceArray<int>(n / 2);
stdgpu::unordered_set<int> set = stdgpu::unordered_set<int>::createDeviceObject(n);

thrust::sequence(stdgpu::device_begin(d_input), stdgpu::device_end(d_input), 1);


thrust::copy_if(stdgpu::device_cbegin(d_input),
stdgpu::device_cend(d_input),
stdgpu::device_begin(d_result),
is_odd());


insert_neighbors(d_result, n / 2, set);


auto range_set = set.device_range();
int sum = thrust::reduce(range_set.begin(), range_set.end(), 0, thrust::plus<int>());

const int sum_closed_form = n * (n + 1) / 2;

std::cout << "The duplicate-free set of numbers contains " << set.size() << " elements (" << n + 1
<< " expected) and the computed sum is " << sum << " (" << sum_closed_form << " expected)" << std::endl;

destroyDeviceArray<int>(d_input);
destroyDeviceArray<int>(d_result);
stdgpu::unordered_set<int>::destroyDeviceObject(set);
}
