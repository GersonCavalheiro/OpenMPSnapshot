#include <complex>
#include <cstdio>
#include <omp.h>
#include "ompvv.h"
const int MR = 16;
const int M = 256;
const int N = 256;
static_assert(!(M % MR), "Block size must be divisible by the matrix size");
namespace ulmBLAS {
template <typename IndexType, typename TL, typename Buffer>
void
trlspack(IndexType   mc,
bool        unit,
const TL    *L,
IndexType   incRowL,
IndexType   incColL,
Buffer      *p)
{
OMPVV_INFOMSG("app_kernel_lsms_triangular_packing");
IndexType mp = (mc+MR-1) / MR;
#pragma omp target teams distribute parallel for collapse(4)
for (IndexType j=0; j<mp; ++j) {
for (IndexType j0=0; j0<MR; ++j0) {
for (IndexType i=j; i<mp; ++i) {
for (IndexType i0=0; i0<MR; ++i0) {
IndexType I  = i*MR+i0;
IndexType J  = j*MR+j0;
IndexType nu = (i+1)*i/2*MR*MR + j*MR*MR + j0*MR +i0;
p[nu] = (I==J && unit)
? Buffer(1)
: (I==J && !unit)
? Buffer(1) / L[I*(incRowL+incColL)]
: (I>=mc || J>=mc)
? Buffer(0)
: (I>J)
? L[I*incRowL+J*incColL]
: Buffer(0);
}
}
}
}
}
} 
int main() {
const bool unit = true;
double A[M*N];
double buffer[M*M + MR];
OMPVV_TEST_OFFLOADING;
for (int i = 0; i < M; i++) 
for (int j = 0; j < N; j++) 
A[i*N + j] = (i == j) ? 1.0 
: (j >  i) ? 0.0
: drand48()*2.0 - 1.0;
#pragma omp target data           map(to:A[0 : M*N])            map(from:buffer[0 : M*M + MR])
{
ulmBLAS::trlspack(M, unit, A, N, 1, buffer);
}
double error_sum = 0.0;
int mp = (M + MR - 1) / MR;
for (int j = 0; j < mp; j++) {
for (int j0 = 0; j0 < MR; j0++) {
for (int i = j; i < mp; i++) {
for (int i0 = 0; i0 < MR; i0++) {
int I  = i*MR+i0;
int J  = j*MR+j0;
int nu = (i+1)*i/2*MR*MR + j*MR*MR + j0*MR +i0;
OMPVV_TEST_AND_SET(error_sum,buffer[nu] != A[I*N + J]);
}
}
}
}
OMPVV_REPORT_AND_RETURN(error_sum != 0.0);
}
