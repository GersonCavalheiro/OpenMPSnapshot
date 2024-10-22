

#include <iostream>
#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>

#include <stdgpu/iterator.h> 
#include <stdgpu/memory.h>   
#include <stdgpu/platform.h> 
#include <stdgpu/vector.cuh> 

void
insert_neighbors_with_duplicates(const int* d_input, const stdgpu::index_t n, stdgpu::vector<int>& vec)
{
#pragma omp parallel for
for (stdgpu::index_t i = 0; i < n; ++i)
{
int num = d_input[i];
int num_neighborhood[3] = { num - 1, num, num + 1 };

for (int num_neighbor : num_neighborhood)
{
vec.push_back(num_neighbor);
}
}
}

int
main()
{

const stdgpu::index_t n = 100;

int* d_input = createDeviceArray<int>(n);
stdgpu::vector<int> vec = stdgpu::vector<int>::createDeviceObject(3 * n);

thrust::sequence(stdgpu::device_begin(d_input), stdgpu::device_end(d_input), 1);


insert_neighbors_with_duplicates(d_input, n, vec);


auto range_vec = vec.device_range();
int sum = thrust::reduce(range_vec.begin(), range_vec.end(), 0, thrust::plus<int>());

const int sum_closed_form = 3 * (n * (n + 1) / 2);

std::cout << "The set of duplicated numbers contains " << vec.size() << " elements (" << 3 * n
<< " expected) and the computed sum is " << sum << " (" << sum_closed_form << " expected)" << std::endl;

destroyDeviceArray<int>(d_input);
stdgpu::vector<int>::destroyDeviceObject(vec);
}
