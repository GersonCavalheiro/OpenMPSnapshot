

#include <iostream>
#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>

#include <stdgpu/deque.cuh>  
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
insert_neighbors_with_duplicates(const int* d_input, const stdgpu::index_t n, stdgpu::deque<int>& deq)
{
#pragma omp parallel for
for (stdgpu::index_t i = 0; i < n; ++i)
{
int num = d_input[i];
int num_neighborhood[3] = { num - 1, num, num + 1 };

is_odd odd;
for (int num_neighbor : num_neighborhood)
{
if (odd(num_neighbor))
{
deq.push_back(num_neighbor);
}
else
{
deq.push_front(num_neighbor);
}
}
}
}

int
main()
{

const stdgpu::index_t n = 100;

int* d_input = createDeviceArray<int>(n);
stdgpu::deque<int> deq = stdgpu::deque<int>::createDeviceObject(3 * n);

thrust::sequence(stdgpu::device_begin(d_input), stdgpu::device_end(d_input), 1);


insert_neighbors_with_duplicates(d_input, n, deq);


auto range_deq = deq.device_range();
int sum = thrust::reduce(range_deq.begin(), range_deq.end(), 0, thrust::plus<int>());

const int sum_closed_form = 3 * (n * (n + 1) / 2);

std::cout << "The set of duplicated numbers contains " << deq.size() << " elements (" << 3 * n
<< " expected) and the computed sum is " << sum << " (" << sum_closed_form << " expected)" << std::endl;

destroyDeviceArray<int>(d_input);
stdgpu::deque<int>::destroyDeviceObject(deq);
}
