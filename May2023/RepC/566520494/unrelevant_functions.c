#include <stdio.h>
#include <stdlib.h>
typedef uint64_t u64;
typedef uint32_t u32;
void transpose(u32 *T, u32 const *A){
#pragma omp parallel for
for(int i = 0 ; i < n ; i++)
for(int j = 0 ; j < n ; j++)
T[j * n + i] = A[i * n + j];
}
void transpose2(u32 *T, int length, int stride){
#pragma omp parallel for
for(int i = 0; i < stride; i++)
for(int j = 0; j < stride; j++)
{      
u32 tmp =  T[i * length + j];
T[i * length + j] = T[j * length + i];
T[j * length + i] = tmp;
}
}
void block_transpose(u32 *T, int length, int stride) {
if(stride <= 32){
transpose2(T, length, stride);
}
else {
if(stride&1) { 
for(int i = 1 ; i < stride ; i++){ 
u32 tmp = T[i];
T[i] = T[i * length];
T[i * length] = tmp;
}
block_transpose(T + length + 1, length, --stride);
}
else { 
int stride0 = stride >> 1;
u32 *tmp_block = (u32*) malloc(sizeof(u32) * stride0);
u32 *block1 = T;
u32 *block2 = T + stride0;
u32 *block3 = T + stride0 * length;
u32 *block4 = T + stride0 * (length+1);
for(int i = 0 ; i < (stride >> 1) ; i++){
memcpy(tmp_block, block2 + i*length,  sizeof(u32) * stride0);
memcpy(block2 + i*length,  block3 + i*length,  sizeof(u32) * stride0);
memcpy(block3 + i*length,  tmp_block, sizeof(u32) * stride0);
}
free(tmp_block);
block_transpose(block1, length, stride0); 
block_transpose(block2, length, stride0); 
block_transpose(block3, length, stride0); 
block_transpose(block4, length, stride0); 
}
}
}
void gemm(long m, long n, long l, long strideU, long strideV, long strideW, 
u32 const *U, u32 const *V, u32 *W)
{
#pragma omp parallel for
for (long i = 0; i < m; i++)
for (long k = 0; k < n; k++)
for (long j = 0; j < l; j++)
{
u64 x = W[i * strideW + j];
u64 y = U[i * strideU + k];
u64 z = V[k * strideV + j]; 
W[i * strideW + j] = (x + y * z) % prime;
}
}
void gemm_rec(long m, long n, long l, long strideU, long strideV, long strideW, u32 const *U, u32 const *V, u32 *W){
if(m <= 32 || n <= 32 || l <= 32) {
return gemm(m, n, l, strideU, strideV, strideW, U, V, W);
}
long m0 = m >> 1;
long n0 = n >> 1;
long l0 = l >> 1;
long m1 = m - m0;
long n1 = n - n0;
long l1 = l - l0;
gemm_rec(m0, n0, l0, strideU, strideV, strideW, U, V, W); 
gemm_rec(m1, n0, l0, strideU, strideV, strideW, U + m0 * strideU, V, W + m0 * strideW); 
gemm_rec(m0, n0, l1, strideU, strideV, strideW, U, V + l0, W + l0); 
gemm_rec(m1, n0, l1, strideU, strideV, strideW, U + m0 * strideU, V + l0, W + m0 * strideW + l0); 
gemm_rec(m0, n1, l0, strideU, strideV, strideW, U + n0, V + n0 * strideV, W); 
gemm_rec(m1, n1, l0, strideU, strideV, strideW, U + m0 * strideU + n0, V + n0 * strideV, W + m0 * strideW); 
gemm_rec(m0, n1, l1, strideU, strideV, strideW, U + n0, V + n0 * strideV + l0, W + l0); 
gemm_rec(m1, n1, l1, strideU, strideV, strideW, U + m0 * strideU + n0, V + n0 * strideV + l0, W + m0 * strideW + l0); 
}
void gemmt(long m, long n, long l, long strideU, long strideV, long strideW, 
u32 const *U, u32 const *V, u32 *W)
{
u32 * Ut = (u32*) malloc(sizeof(u32) * m * n);
memcpy(Ut, U, sizeof(u32) * m * n);
block_transpose(Ut, m, m);
gemm_rec(m, n, l, strideU, strideV, strideW, Ut, V, W);
free(Ut);
}
