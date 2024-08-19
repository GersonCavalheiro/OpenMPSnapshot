
#include "matrix.h"

bool Matrix::multiplyCPU_STD(Matrix *calculated, GPU *gpu) {

for (int i = 0; i < row; i++) {

for (int j = 0; j < B->col; j++) {

for (int k = 0; k < col; k++) {

calculated->mem[i * B->col + j] +=
mem[i * col + k] * B->mem[k * B->col + j];
}
}
}

return true;
}

bool Matrix::multiplyCPU_TILED(Matrix *calculated, GPU *gpu) {

for (int i = 0; i < B->col; i += TILESIZE) {

for (int k = 0; k < col; k += TILESIZE) {

for (int j = 0; j < row; j++) {

for (int kk = k; kk < k + TILESIZE; kk++) {

for (int ii = i; ii < i + TILESIZE; ii++) {

calculated->mem[j * B->col + ii] +=
mem[j * col + kk] * B->mem[kk * B->col + ii];
}
}
}
}
}

return true;
}

bool Matrix::multiplyCPU_TILED_BASIC(Matrix *calculated, GPU *gpu) {

for (int j0 = 0; j0 < row; j0 += TILESIZE) {

for (int i0 = 0; i0 < B->col; i0 += TILESIZE) {

for (int k0 = 0; k0 < col; k0 += TILESIZE) {

for (int j = j0; j < j0 + TILESIZE; j++) {

for (int i = i0; i < i0 + TILESIZE; i++) {

for (int k = k0; k < k0 + TILESIZE; k++) {

calculated->mem[j * B->col + i] +=
mem[j * col + k] * B->mem[k * B->col + i];
}
}
}
}
}
}

return true;
}

bool Matrix::multiplyCPU_TILED_OMP(Matrix *calculated, GPU *gpu) {

#pragma omp parallel for
for (int i = 0; i < B->col; i += TILESIZE) {
#pragma omp parallel for
for (int k = 0; k < col; k += TILESIZE) {
#pragma omp parallel for
for (int j = 0; j < row; j++) {

for (int kk = k; kk < k + TILESIZE; kk++) {

for (int ii = i; ii < i + TILESIZE; ii++) {

calculated->mem[j * B->col + ii] +=
mem[j * col + kk] * B->mem[kk * B->col + ii];
}
}
}
}
}

return true;
}
#ifdef __OPENCL__
bool Matrix::multiplyGPU_STD(Matrix *calculated, GPU *gpu) {

return multiplyGPU(calculated, MULTYPE_GPU_STD, gpu);
}

bool Matrix::multiplyGPU_VEC4(Matrix *calculated, GPU *gpu) {

return multiplyGPU(calculated, MULTYPE_GPU_VEC4, gpu);
}

bool Matrix::multiplyGPU_VEC8(Matrix *calculated, GPU *gpu) {

return multiplyGPU(calculated, MULTYPE_GPU_VEC8, gpu);
}

#ifndef __ARM__
bool Matrix::multiplyGPU_DISC(Matrix *calculated, GPU *gpu) {

return multiplyGPU(calculated, MULTYPE_GPU_DISC, gpu);
}
#endif

bool Matrix::multiplyGPU(Matrix *calculated, int type, GPU *gpu) {

if (!gpu->getEnabled()) {
return false;
}

cl_int errCode;

size_t localWorkSize[2], globalWorkSize[2];

globalWorkSize[0] = B->col / funcList[type].argument[0];
globalWorkSize[1] = row / funcList[type].argument[1];

localWorkSize[0] = (size_t) funcList[type].argument[2];
localWorkSize[1] = (size_t) funcList[type].argument[3];

#ifndef __ARM__
errCode = clEnqueueWriteBuffer(gpu->clCommandQue, buf_mem, CL_FALSE, 0, mem_size, mem, 0, NULL, NULL);
gpu->checkErr("clEnqueueWriteBuffer", errCode);

errCode = clEnqueueWriteBuffer(gpu->clCommandQue, B->buf_mem, CL_FALSE, 0, mem_size, B->mem, 0, NULL, NULL);
gpu->checkErr("clEnqueueWriteBuffer", errCode);
#else
errCode = clEnqueueUnmapMemObject(gpu->clCommandQue, buf_mem, mem, 0, NULL, NULL);
errCode = clEnqueueUnmapMemObject(gpu->clCommandQue, B->buf_mem, B->mem, 0, NULL, NULL);
errCode = clEnqueueUnmapMemObject(gpu->clCommandQue, calculated->buf_mem, calculated->mem, 0, NULL, NULL);
#endif

errCode = clSetKernelArg(funcList[type].kernels[0], 0, sizeof(cl_mem), (void *) &buf_mem);
errCode |= clSetKernelArg(funcList[type].kernels[0], 1, sizeof(cl_mem), (void *) &B->buf_mem);
errCode |= clSetKernelArg(funcList[type].kernels[0], 2, sizeof(cl_mem), (void *) &calculated->buf_mem);
errCode |= clSetKernelArg(funcList[type].kernels[0], 3, sizeof(int), (void *) &col);
errCode |= clSetKernelArg(funcList[type].kernels[0], 4, sizeof(int), (void *) &B->col);
gpu->checkErr("clSetKernelArg", errCode);

errCode = clEnqueueNDRangeKernel(gpu->clCommandQue, funcList[type].kernels[0], 2,
NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
gpu->checkErr("clEnqueueNDRangeKernel", errCode);

#ifndef __ARM__
errCode = clEnqueueReadBuffer(gpu->clCommandQue, calculated->buf_mem, CL_TRUE, 0,
calculated->mem_size, calculated->mem, 0, NULL, NULL);
gpu->checkErr("clEnqueueReadBuffer", errCode);
#else
mem = (float *) clEnqueueMapBuffer(gpu->clCommandQue, buf_mem, CL_TRUE,
CL_MAP_READ | CL_MAP_WRITE, 0, mem_size, 0, NULL, NULL, &errCode);

B->mem = (float *) clEnqueueMapBuffer(gpu->clCommandQue, B->buf_mem, CL_TRUE,
CL_MAP_READ | CL_MAP_WRITE, 0, mem_size, 0, NULL, NULL, &errCode);

calculated->mem = (float *) clEnqueueMapBuffer(gpu->clCommandQue, calculated->buf_mem, CL_TRUE,
CL_MAP_READ | CL_MAP_WRITE, 0, mem_size, 0, NULL, NULL, &errCode);
#endif

clFinish(gpu->clCommandQue);

return true;
}
#endif
