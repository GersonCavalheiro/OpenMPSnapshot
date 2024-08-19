
#include "conv.h"

bool Conv::convCPU_STD(Conv *calculated, GPU *gpu) {

int half_length = (filter_length - 1) / 2;

for(int j = 0; j < row; j++) {

for(int i=(half_length+1); i<(col-(half_length+1)); i++)
{
float acc = 0.0f;
for(int k=0; k<filter_length; k++)
{
acc += mem[j*col + (i+k-half_length)] * filter[k];
}
temp[j*col + i] = acc;
}
}

for(int j=(half_length+1); j<(row-(half_length+1)); j++)
{
for(int i=0; i<col; i++)
{
float acc = 0.0f;
for(int k=0; k<filter_length; k++)
{
acc += temp[(j+k-half_length)*col + i] * filter[k];
}
calculated->mem[j*col + i] = acc;
}
}

return true;
}

bool Conv::convCPU_OMP(Conv *calculated, GPU *gpu) {

int half_length = (filter_length - 1) / 2;

float *local_mem = mem;
float *local_calcmem = calculated->mem;
float *local_temp = temp;
float *local_filter = filter;

#pragma omp parallel for shared(local_mem, local_temp) firstprivate(local_filter) schedule(static)
for(int j=0; j<row; j++)
{
for(int i=(half_length+1); i<(col-(half_length+1)); i++)
{
float acc = 0.0f;
for(int k=0; k<filter_length; k++)
{
acc += local_mem[j*col + (i+k-half_length)] * local_filter[k];
}
local_temp[j*col + i] = acc;
}
}

#pragma omp parallel for shared(local_calcmem, local_temp) firstprivate(local_filter) schedule(static)
for(int j=(half_length+1); j<(row-(half_length+1)); j++)
{
for(int i=0; i<col; i++)
{
float acc = 0.0f;
for(int k=0; k<filter_length; k++)
{
acc += local_temp[(j+k-half_length)*col + i] * local_filter[k];
}
local_calcmem[j*col + i] = acc;
}
}

return true;
}

#ifdef __OPENCL__
bool Conv::convGPU_STD(Conv *calculated, GPU *gpu) {

return convGPU(calculated, CONVTYPE_GPU_STD, gpu);
}

bool Conv::convGPU_VEC4(Conv *calculated, GPU *gpu) {

return convGPU(calculated, CONVTYPE_GPU_VEC4, gpu);
}

bool Conv::convGPU_COMB(Conv *calculated, GPU *gpu) {

if (!gpu->getEnabled()) {
return false;
}

cl_int errCode;

cl_mem buf_filter = clCreateBuffer(gpu->clGPUContext,
CL_MEM_READ_ONLY,
(size_t) (filter_length * sizeof(float)),
NULL,
&errCode);

size_t  localWorkSize[2], globalWorkSize[2];

#ifndef __ARM__
errCode = clEnqueueWriteBuffer(gpu->clCommandQue, buf_mem, CL_FALSE, 0, mem_size, mem, 0, NULL, NULL);
gpu->checkErr("clEnqueueWriteBuffer", errCode);
#else
errCode = clEnqueueUnmapMemObject(gpu->clCommandQue, buf_mem, mem, 0, NULL, NULL);
gpu->checkErr("clEnqueueUnmapMemObject, mem", errCode);
errCode = clEnqueueUnmapMemObject(gpu->clCommandQue, calculated->buf_mem, calculated->mem, 0, NULL, NULL);
gpu->checkErr("clEnqueueUnmapMemObject, calcmem", errCode);
errCode = clEnqueueUnmapMemObject(gpu->clCommandQue, buf_temp, temp, 0, NULL, NULL);
gpu->checkErr("clEnqueueUnmapMemObject, temp", errCode);
#endif

errCode = clEnqueueWriteBuffer(gpu->clCommandQue, buf_filter, CL_FALSE, 0, filter_length * sizeof(float), filter, 0, NULL, NULL);
gpu->checkErr("clEnqueueWriteBuffer", errCode);

errCode  = clSetKernelArg(funcList[CONVTYPE_GPU_COMB].kernels[0], 0, sizeof(cl_mem), (void*)&calculated->buf_mem);
errCode |= clSetKernelArg(funcList[CONVTYPE_GPU_COMB].kernels[0], 1, sizeof(cl_mem), (void*)&buf_mem);
errCode |= clSetKernelArg(funcList[CONVTYPE_GPU_COMB].kernels[0], 2, sizeof(cl_mem), (void*)&buf_filter);
errCode |= clSetKernelArg(funcList[CONVTYPE_GPU_COMB].kernels[0], 3, sizeof(int),    (void*)&col);
errCode |= clSetKernelArg(funcList[CONVTYPE_GPU_COMB].kernels[0], 4, sizeof(int),    (void*)&row);
errCode |= clSetKernelArg(funcList[CONVTYPE_GPU_COMB].kernels[0], 5, sizeof(int),    (void*)&col);
gpu->checkErr("clSetKernelArg", errCode);

localWorkSize[0] = ROWS_BLOCKDIM_X;
localWorkSize[1] = ROWS_BLOCKDIM_Y;

globalWorkSize[0] = col / 4;
globalWorkSize[1] = row;

errCode = clEnqueueNDRangeKernel(gpu->clCommandQue, funcList[CONVTYPE_GPU_COMB].kernels[0], 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
gpu->checkErr("clEnqueueNDRangeKernel", errCode);


#ifndef __ARM__
errCode = clEnqueueReadBuffer(gpu->clCommandQue, calculated->buf_mem, CL_TRUE, 0, mem_size, calculated->mem, 0, NULL, NULL);
gpu->checkErr("clEnqueueReadBuffer", errCode);
#else
mem = (float *) clEnqueueMapBuffer(gpu->clCommandQue, buf_mem, CL_TRUE,
CL_MAP_READ | CL_MAP_WRITE, 0, mem_size, 0, NULL, NULL, &errCode);
gpu->checkErr("clEnqueueMapBuffer2, mem", errCode);

temp = (float *) clEnqueueMapBuffer(gpu->clCommandQue, buf_temp, CL_TRUE,
CL_MAP_READ | CL_MAP_WRITE, 0, mem_size, 0, NULL, NULL, &errCode);
gpu->checkErr("clEnqueueMapBuffer2, temp", errCode);

calculated->mem = (float *) clEnqueueMapBuffer(gpu->clCommandQue, calculated->buf_mem, CL_TRUE,
CL_MAP_READ | CL_MAP_WRITE, 0, mem_size, 0, NULL, NULL, &errCode);
gpu->checkErr("clEnqueueMapBuffer2, calcmem", errCode);
#endif

clFinish(gpu->clCommandQue);

clReleaseMemObject(buf_filter);

return true;
}

bool Conv::convGPU(Conv *calculated, int type, GPU *gpu) {

if (!gpu->getEnabled()) {
return false;
}

cl_int errCode;

cl_mem buf_filter = clCreateBuffer(gpu->clGPUContext,
CL_MEM_READ_ONLY,
(size_t) (filter_length * sizeof(float)),
NULL,
&errCode);

size_t  localWorkSizeRows[2], globalWorkSizeRows[2];
size_t  localWorkSizeCols[2], globalWorkSizeCols[2];

#ifndef __ARM__
errCode = clEnqueueWriteBuffer(gpu->clCommandQue, buf_mem, CL_FALSE, 0, mem_size, mem, 0, NULL, NULL);
gpu->checkErr("clEnqueueWriteBuffer", errCode);
#else
errCode = clEnqueueUnmapMemObject(gpu->clCommandQue, buf_mem, mem, 0, NULL, NULL);
gpu->checkErr("clEnqueueUnmapMemObject, mem", errCode);
errCode = clEnqueueUnmapMemObject(gpu->clCommandQue, calculated->buf_mem, calculated->mem, 0, NULL, NULL);
gpu->checkErr("clEnqueueUnmapMemObject, calcmem", errCode);
errCode = clEnqueueUnmapMemObject(gpu->clCommandQue, buf_temp, temp, 0, NULL, NULL);
gpu->checkErr("clEnqueueUnmapMemObject, temp", errCode);
#endif

errCode = clEnqueueWriteBuffer(gpu->clCommandQue, buf_filter, CL_FALSE, 0, filter_length * sizeof(float), filter, 0, NULL, NULL);
gpu->checkErr("clEnqueueWriteBuffer", errCode);

errCode  = clSetKernelArg(funcList[type].kernels[0], 0, sizeof(cl_mem), (void*)&buf_temp);
errCode |= clSetKernelArg(funcList[type].kernels[0], 1, sizeof(cl_mem), (void*)&buf_mem);
errCode |= clSetKernelArg(funcList[type].kernels[0], 2, sizeof(cl_mem), (void*)&buf_filter);
errCode |= clSetKernelArg(funcList[type].kernels[0], 3, sizeof(int),    (void*)&col);
errCode |= clSetKernelArg(funcList[type].kernels[0], 4, sizeof(int),    (void*)&row);
errCode |= clSetKernelArg(funcList[type].kernels[0], 5, sizeof(int),    (void*)&col);
gpu->checkErr("clSetKernelArg", errCode);

localWorkSizeRows[0] = ROWS_BLOCKDIM_X;
localWorkSizeRows[1] = ROWS_BLOCKDIM_Y;

globalWorkSizeRows[0] = col / funcList[type].argument[0];
globalWorkSizeRows[1] = row;

errCode = clEnqueueNDRangeKernel(gpu->clCommandQue, funcList[type].kernels[0], 2, NULL, globalWorkSizeRows, localWorkSizeRows, 0, NULL, NULL);
gpu->checkErr("clEnqueueNDRangeKernel", errCode);


errCode  = clSetKernelArg(funcList[type].kernels[1], 0, sizeof(cl_mem), (void*)&calculated->buf_mem);
errCode |= clSetKernelArg(funcList[type].kernels[1], 1, sizeof(cl_mem), (void*)&buf_temp);
errCode |= clSetKernelArg(funcList[type].kernels[1], 2, sizeof(cl_mem), (void*)&buf_filter);
errCode |= clSetKernelArg(funcList[type].kernels[1], 3, sizeof(int),    (void*)&col);
errCode |= clSetKernelArg(funcList[type].kernels[1], 4, sizeof(int),    (void*)&row);
errCode |= clSetKernelArg(funcList[type].kernels[1], 5, sizeof(int),    (void*)&col);
gpu->checkErr("clSetKernelArg", errCode);

localWorkSizeCols[0] = COLUMNS_BLOCKDIM_X;
localWorkSizeCols[1] = COLUMNS_BLOCKDIM_Y;

globalWorkSizeCols[0] = col / funcList[type].argument[0];
globalWorkSizeCols[1] = row;

errCode = clEnqueueNDRangeKernel(gpu->clCommandQue, funcList[type].kernels[1], 2, NULL, globalWorkSizeCols, localWorkSizeCols, 0, NULL, NULL);
gpu->checkErr("clEnqueueNDRangeKernel", errCode);

#ifndef __ARM__
errCode = clEnqueueReadBuffer(gpu->clCommandQue, calculated->buf_mem, CL_TRUE, 0, mem_size, calculated->mem, 0, NULL, NULL);
gpu->checkErr("clEnqueueReadBuffer", errCode);

#else
clFinish(gpu->clCommandQue);

mem = (float *) clEnqueueMapBuffer(gpu->clCommandQue, buf_mem, CL_TRUE,
CL_MAP_READ | CL_MAP_WRITE, 0, mem_size, 0, NULL, NULL, &errCode);
gpu->checkErr("clEnqueueMapBuffer, mem", errCode);

temp = (float *) clEnqueueMapBuffer(gpu->clCommandQue, buf_temp, CL_TRUE,
CL_MAP_READ | CL_MAP_WRITE, 0, mem_size, 0, NULL, NULL, &errCode);
gpu->checkErr("clEnqueueMapBuffer, temp", errCode);

calculated->mem = (float *) clEnqueueMapBuffer(gpu->clCommandQue, calculated->buf_mem, CL_TRUE,
CL_MAP_READ | CL_MAP_WRITE, 0, mem_size, 0, NULL, NULL, &errCode);
gpu->checkErr("clEnqueueMapBuffer, calcmem", errCode);
#endif

clReleaseMemObject(buf_filter);

return true;
}
#endif
