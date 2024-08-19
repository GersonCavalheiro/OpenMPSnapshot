#include<iostream>
#include<chrono>
#include<ctime>
#include<CL/sycl.hpp>

using namespace cl::sycl;
using namespace std;

class linear_algebra;
template<typename type>
type*  matmul(sycl::queue& queue, type* A, type* B, int* size)
{
auto device = queue.get_device();
cout << device.get_info<sycl::info::device::name> << endl;
type* C = new type[size[0] * size[1]];

range<1> dim(size[0] * size[1]);
property_list prop = { property::buffer::use_host_ptr() };

buffer<type> buffer_A(A, dim, prop);
buffer<type> buffer_B(B, dim, prop);
buffer<type> out_buffer(C, dim, prop);
queue.submit([&](handler& cgh) {
auto a = buffer_A.template get_access<access::mode::read>(cgh);
auto b = buffer_B.template get_access<access::mode::read>(cgh);
auto out = out_buffer.template get_access<access::mode::write>(cgh);
auto localrange = range<1>(device.get_info<cl::sycl::info::device::max_work_group_size>());
accessor<type, 1, access::mode::read_write, access::target::local> pBA(localrange, cgh);
accessor<type, 1, access::mode::read_write, access::target::local> pBB(localrange, cgh);
int block_size = device.get_info<cl::sycl::info::device::max_work_group_size>())
cgh.parallel_for(linear_algebra)(
nd_range<2>{range<2>(size[0], isze[1]),
range<2>(size[0], size[1])},
[=](nd_item<2> block)
{
int x_block = block.get_group[1];
int y_block = block.get_group[0];

int x_local = block.get_local_id(1);
int y_local = block.get_local_id(0);

int a_start = size[0] * block_size * y_block;
int a_end = a_start + size[0] - 1;

int b_start = block_size * x_block;

type result = 0.0f;

for (int i = a_start, j = b_start;
i <= a_end;
i += block_size, j += (block_size * size[0]))
{
pBA[y_local * block_size + x_local] = a[i + size[0] * y_local + x_local];

pBB[x_local * block_size + y_local] = b[j + size[0] * y_local + x_local];
block.barrier(access::fence_space::local_space);
for (int k = 0; k < block_size;k++)
{
result += pBA[y_local * block_size + k] * pBB[x_local * block_size + k];

}
block.barrier(access::fence_space::local_space);
}
auto index = block.get_global_id(0) * block.get_global_range()[1] + block.get_global_id(1);
out_buffer[index] = result;
)
});
});
return C;
}


void matmul(float* A, float* B, float* C, int m, int n, int p){
#pragma omp parallel for
for(int i=0; i<m; i++){
for(int j=0; j<p; j++){
float result = 0.0f;
#pragma omp simd
for(int k=0; k<n; k++){
result += *(A + i*m + k) * *(B + k*p + j);
}
*(C + i*m + j) = result;
}
}
}
