
#include "scan.h"

unsigned int Scan::iSnapUp(unsigned int dividend, unsigned int divisor) {

return ((dividend % divisor) == 0) ? dividend : (dividend - dividend % divisor + divisor);
}

bool Scan::scanCPU_STD(Scan *calculated, GPU *gpu) {

float sum = 0;
for (int i = 0; i < size; i++) {
sum += mem[i];
calculated->mem[i] = sum;
}

return true;
}
#ifndef __ARM__
#ifdef __AVX__
inline __m256 scan_AVX(__m256 x) {
__m256 t0, t1;
t0 = _mm256_permute_ps(x, _MM_SHUFFLE(2, 1, 0, 3));
t1 = _mm256_permute2f128_ps(t0, t0, 41);
x = _mm256_add_ps(x, _mm256_blend_ps(t0, t1, 0x11));
t0 = _mm256_permute_ps(x, _MM_SHUFFLE(1, 0, 3, 2));
t1 = _mm256_permute2f128_ps(t0, t0, 41);
x = _mm256_add_ps(x, _mm256_blend_ps(t0, t1, 0x33));
x = _mm256_add_ps(x,_mm256_permute2f128_ps(x, x, 41));
return x;
}

bool Scan::scanCPU_AVX(Scan *calculated, GPU *gpu) {

__m256 offset = _mm256_setzero_ps();

for (int i = 0; i < size; i += 8) {
__m256 x = _mm256_loadu_ps(&mem[i]);
__m256 out = scan_AVX(x);
out = _mm256_add_ps(out, offset);
_mm256_storeu_ps(&calculated->mem[i], out);
__m256 t0 = _mm256_permute2f128_ps(out, out, 0x11);
offset = _mm256_permute_ps(t0, 0xff);
}

return true;
}
#endif

inline __m128 scan_SSE(__m128 x) {

x = _mm_add_ps(x, _mm_castsi128_ps(_mm_slli_si128(_mm_castps_si128(x), 4)));
x = _mm_add_ps(x, _mm_castsi128_ps(_mm_slli_si128(_mm_castps_si128(x), 8)));
return x;
}

bool Scan::scanCPU_SSE(Scan *calculated, GPU *gpu) {

__m128 offset = _mm_setzero_ps();

for (int i = 0; i < size; i += 4)	{

__m128 x = _mm_load_ps(&mem[i]);
__m128 out = scan_SSE(x);
out = _mm_add_ps(out, offset);
_mm_store_ps(&calculated->mem[i], out);
offset = _mm_shuffle_ps(out, out, _MM_SHUFFLE(3, 3, 3, 3));
}

return true;
}

bool Scan::scanCPU_OMP_SSE(Scan *calculated, GPU *gpu) {

float *suma;
#pragma omp parallel
{
const int ithread = omp_get_thread_num();
const int nthreads = omp_get_num_threads();
#pragma omp single
{
suma = new float[nthreads + 1];
suma[0] = 0;
}
float sum = 0;
#pragma omp for schedule(static) nowait 
for (int i = 0; i < size; i++) {
sum += mem[i];
calculated->mem[i] = sum;
}
suma[ithread + 1] = sum;
#pragma omp barrier
#pragma omp single
{
float tmp = 0;
for (int i = 0; i<(nthreads + 1); i++) {
tmp += suma[i];
suma[i] = tmp;
}
}
__m128 offset = _mm_set1_ps(suma[ithread]);
#pragma omp for schedule(static) 
for (int i = 0; i < size / 4; i++) {
__m128 tmp = _mm_load_ps(&calculated->mem[4*i]);
tmp = _mm_add_ps(tmp, offset);
_mm_store_ps(&calculated->mem[4*i], tmp);
}
}
delete[] suma;

return true;
}

#endif

#ifdef __OPENCL__
bool Scan::scanGPU_STD(Scan *calculated, GPU *gpu) {

return scanGPU(calculated, SCANTYPE_GPU_STD, gpu);
}

bool Scan::scanGPU(Scan *calculated, int type, GPU *gpu) {

if (!gpu->getEnabled()) {
return false;
}

cl_int errCode;

unsigned int local1size = (4 * WORKGROUP_SIZE);
unsigned int elements = (uint32_t) size / local1size;
unsigned int local2size = ARRAY_LENGTH / local1size;

cl_mem buf_temp = clCreateBuffer(gpu->clGPUContext,
CL_MEM_READ_WRITE,
(size_t) (mem_size),
NULL,
&errCode);

#ifndef __ARM__
errCode = clEnqueueWriteBuffer(gpu->clCommandQue, buf_mem, CL_FALSE, 0, mem_size, mem, 0, NULL, NULL);
gpu->checkErr("clEnqueueWriteBuffer", errCode);
#else
errCode = clEnqueueUnmapMemObject(gpu->clCommandQue, buf_mem, mem, 0, NULL, NULL);
errCode = clEnqueueUnmapMemObject(gpu->clCommandQue, calculated->buf_mem, calculated->mem, 0, NULL, NULL);
#endif

errCode  = clSetKernelArg(funcList[type].kernels[0], 0, sizeof(cl_mem), (void *)&calculated->buf_mem);
errCode |= clSetKernelArg(funcList[type].kernels[0], 1, sizeof(cl_mem), (void *)&buf_mem);
errCode |= clSetKernelArg(funcList[type].kernels[0], 2, 2 * WORKGROUP_SIZE * sizeof(float), NULL);
errCode |= clSetKernelArg(funcList[type].kernels[0], 3, sizeof(unsigned int), (void *)&local1size);
gpu->checkErr("clSetKernelArg", errCode);

size_t  localWorkSize1 = WORKGROUP_SIZE;
size_t  globalWorkSize1 = size / 4;

errCode = clEnqueueNDRangeKernel(gpu->clCommandQue, funcList[type].kernels[0], 1,
NULL, &globalWorkSize1, &localWorkSize1, 0, NULL, NULL);
gpu->checkErr("clEnqueueNDRangeKernel", errCode);


errCode  = clSetKernelArg(funcList[type].kernels[1], 0, sizeof(cl_mem), (void *)&buf_temp);
errCode |= clSetKernelArg(funcList[type].kernels[1], 1, sizeof(cl_mem), (void *)&calculated->buf_mem);
errCode |= clSetKernelArg(funcList[type].kernels[1], 2, sizeof(cl_mem), (void *)&buf_mem);
errCode |= clSetKernelArg(funcList[type].kernels[1], 3, 2 * WORKGROUP_SIZE * sizeof(float), NULL);
errCode |= clSetKernelArg(funcList[type].kernels[1], 4, sizeof(unsigned int), (void *)&elements);
errCode |= clSetKernelArg(funcList[type].kernels[1], 5, sizeof(unsigned int), (void *)&local2size);
gpu->checkErr("clSetKernelArg", errCode);

size_t  localWorkSize2 = WORKGROUP_SIZE;
size_t  globalWorkSize2 = iSnapUp(elements, WORKGROUP_SIZE);

errCode = clEnqueueNDRangeKernel(gpu->clCommandQue, funcList[type].kernels[1], 1, NULL, &globalWorkSize2, &localWorkSize2, 0, NULL, NULL);
gpu->checkErr("clEnqueueNDRangeKernel", errCode);

errCode  = clSetKernelArg(funcList[type].kernels[2], 0, sizeof(cl_mem), (void *)&calculated->buf_mem);
errCode |= clSetKernelArg(funcList[type].kernels[2], 1, sizeof(cl_mem), (void *)&buf_temp);
gpu->checkErr("clSetKernelArg", errCode);

size_t localWorkSize3 = WORKGROUP_SIZE;
size_t globalWorkSize3 = elements * WORKGROUP_SIZE;

errCode = clEnqueueNDRangeKernel(gpu->clCommandQue, funcList[type].kernels[2], 1, NULL, &globalWorkSize3, &localWorkSize3, 0, NULL, NULL);
gpu->checkErr("clEnqueueNDRangeKernel", errCode);

#ifndef __ARM__
errCode = clEnqueueReadBuffer(gpu->clCommandQue, calculated->buf_mem, CL_TRUE, 0, mem_size, calculated->mem, 0, NULL, NULL);
gpu->checkErr("clEnqueueReadBuffer", errCode);
#else
mem = (float *) clEnqueueMapBuffer(gpu->clCommandQue, buf_mem, CL_TRUE,
CL_MAP_READ | CL_MAP_WRITE, 0, mem_size, 0, NULL, NULL, &errCode);

calculated->mem = (float *) clEnqueueMapBuffer(gpu->clCommandQue, calculated->buf_mem, CL_TRUE,
CL_MAP_READ | CL_MAP_WRITE, 0, mem_size, 0, NULL, NULL, &errCode);
#endif

clFinish(gpu->clCommandQue);

clReleaseMemObject(buf_temp);

return true;
}
#endif
