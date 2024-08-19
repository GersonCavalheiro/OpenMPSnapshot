#include <iostream>
#include <vector>
#include <omp.h>

using n_t = float;

const int BlockSize = 16;
struct dim3 { int x{}, y{}; };

# define __shared__ static 
# define __global__ 
void barrier() {
#pragma omp barrier
}
# define __syncthreads() barrier()

template<unsigned int BlockSize>
__global__ void matrix_multiplication_kernel(
const dim3 & gridDim, const dim3 & blockDim, const dim3 & blockIdx, const dim3 & threadIdx,
const n_t*A, const n_t*B, n_t*C, const unsigned int a_cols)
{
const unsigned int tx = threadIdx.x;
const unsigned int ty = threadIdx.y;
const unsigned int bx = blockIdx.x;
const unsigned int by = blockIdx.y;

const unsigned int b_cols = blockDim.x * gridDim.x;

const unsigned int steps = a_cols / BlockSize;

float thread_result = 0.0F;
for(unsigned int step = 0; step < steps; step++)
{
__shared__ float a_values[BlockSize][BlockSize];
__shared__ float b_values[BlockSize][BlockSize];

const unsigned int a_idx = BlockSize * (a_cols * by + step);

const unsigned int b_idx = BlockSize * (b_cols * step + bx);

a_values[ty][tx] = A[a_idx + a_cols * ty + tx];
b_values[ty][tx] = B[b_idx + b_cols * ty + tx];

__syncthreads();

for(unsigned int i = 0; i < BlockSize; i++)
{
thread_result += a_values[ty][i] * b_values[i][tx];
}

__syncthreads();
}

const unsigned block_offset = b_cols * BlockSize * by + BlockSize * bx;

C[block_offset + b_cols * ty + tx] = thread_result;
}

template<typename F, typename... Ts>
void launch2D(const dim3 & numBlocks, const dim3 & blockDim, F & f, Ts&&... ts)
{
for (int bx=0;bx<numBlocks.x;++bx)
for (int by=0;by<numBlocks.y;++by)
{
#pragma omp parallel num_threads(blockDim.x*blockDim.y)
{
const int tn = omp_get_thread_num();
const int tx = tn % blockDim.y;
const int ty = tn / blockDim.y;
f(numBlocks, blockDim, {bx,by}, {tx,ty}, ts...);
}
}
}

int main()
{
const int m{7}; 
const int N{BlockSize*m};
const std::vector<n_t> a(N*N,2.0);
const std::vector<n_t> b(N*N,2.0);
std::vector<n_t> c(N*N,0);
const dim3 threadsperBlock {BlockSize,BlockSize};
const dim3 numBlocks{N/threadsperBlock.x,N/threadsperBlock.y};
launch2D(numBlocks, threadsperBlock, matrix_multiplication_kernel<BlockSize>, a.data(), b.data(), c.data(), N);

int wv = 0;
for(int i=0; i<N; ++i)
for(int j=0; j<N; ++j)
{
n_t c_ij = 0;
for(int k=0; k<N; ++k)
c_ij += a[i*N+k] * b[k*N+j];
if ( (c_ij != c[i*N+j]))
{
wv += (c_ij != c[i*N+j]);
}
}
if(wv)
std::cout << wv << " wrong values! of " << N * N << std::endl;
else
std::cout << "all OK" << std::endl;
return wv != 0;
}

