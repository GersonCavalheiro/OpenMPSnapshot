#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <multMV.h>
#define IND_mult(x, y, z) ((x) + (y)*nx + (z)*ny*nx)
void initSpMat(SpMatrix *mat, int nz, int nRows) {
mat->nz = nz;
mat->nRows = nRows;
mat->value = (TYPE *) aligned_alloc(32, sizeof(TYPE) * nz);
mat->col = (int *) aligned_alloc(32, sizeof(int) * nz);
mat->rowIndex = (int *) calloc(nRows + 1, sizeof(int));
}
void freeSpMat(SpMatrix *mat) {
if (mat) {
free(mat->value);
free(mat->col);
free(mat->rowIndex);
}
}
inline void copyingBorders(TYPE *vec, int nx, int ny, int nz) {
memcpy(vec + IND_mult(0, 0, 0), vec + IND_mult(0, 0, 1), nx * ny * sizeof(TYPE));
for (int z = 1; z < nz - 1; z++) {
memcpy(vec + IND_mult(0, 0, z), vec + IND_mult(0, 1, z), nx * sizeof(TYPE));
for (int y = 1; y < ny - 1; y++) {
vec[IND_mult(0, y, z)] = vec[IND_mult(1, y, z)];
vec[IND_mult(nx - 1, y, z)] = vec[IND_mult(nx - 2, y, z)];
} 
memcpy(vec + IND_mult(0, ny - 1, z), vec + IND_mult(0, ny - 2, z), nx * sizeof(TYPE));
} 
memcpy(vec + IND_mult(0, 0, nz - 1), vec + IND_mult(0, 0, nz - 2), nx * ny * sizeof(TYPE));
}
void multMV_default(TYPE *result, SpMatrix mat, TYPE *vec) {
TYPE localSum;
#pragma omp parallel private(localSum) if (ENABLE_PARALLEL)
{
#pragma omp for nowait
for (int i = 0; i < mat.nRows; i++) {
localSum = 0.0;
for (int j = mat.rowIndex[i]; j < mat.rowIndex[i + 1]; j++)
localSum += mat.value[j] * vec[mat.col[j]];
result[i] = localSum;
}
}
}
#if AVX2_RUN
inline void multMV_AVX_v2(TYPE* result, SpMatrix mat, TYPE* vec, int nx, int ny, int nz) {
__m256d mask = _mm256_setr_pd(-1,-1,-1, 0);
__m256d zero =_mm256_setzero_pd();
copyingBorders(vec, nx, ny, nz);
#pragma omp parallel for if (ENABLE_PARALLEL)
for (int z = 1; z < nz - 1; z++) {
for (int y = 1; y < ny - 1; y++) {
for (int x = 1; x < nx - 1; x++) {
int *col = &mat.col[IND_mult(x,y,z)*NR];
TYPE * val = &mat.value[IND_mult(x,y,z)*NR];
m_ind v_col = mm_load_si(col);
m_real v_val = mm_load(val);
m_real v_vec = mm_i32gather(vec, v_col);
m_real rowsum = mm_mul(v_vec, v_val);
v_col = mm_load_si(col + LENVEC);
v_val = mm_load(val + LENVEC);
v_vec = _mm256_mask_i32gather_pd(zero, vec, v_col, mask, 8);
rowsum = mm_fmadd(v_vec, v_val, rowsum);
rowsum = mm_hadd(rowsum, rowsum);
result[IND_mult(x,y,z)] = ((TYPE*)&rowsum)[0] + ((TYPE*)&rowsum)[2];
} 
} 
} 
}
inline void multMV_AVX_1(TYPE* result, SpMatrix mat, TYPE* vec, int nx, int ny, int nz) {
TYPE * val = mat.value;
TYPE *vec2 = vec;
copyingBorders(vec, nx, ny, nz);
val += nx*ny * NR;   result += nx*ny;   vec2 += nx*ny;
#pragma omp parallel for if (ENABLE_PARALLEL)
for (int z = 1; z < nz - 1; z++) {
val += nx * NR;   result += nx;   vec2 += nx;
for (int y = 1; y < ny - 1; y++) {
val += 1 * NR;   result += 1;   vec2 += 1;
for (int x = 1; x < nx - 1; x += LENVEC) {
m_real vec4_z_l = mm_load(vec2 - nx*ny);
m_ind index4 = mm_set_epi32(0);
m_real val4 = mm_i32gather(val, index4);
m_real sum4 = _mm256_mul_pd(val4, vec4_z_l);
m_real vec4_y_l = mm_load(vec2 - nx);
index4 = mm_set_epi32(1);
val4 = mm_i32gather(val, index4);
sum4 = mm_fmadd(val4, vec4_y_l, sum4);
m_real vec4_x_l = mm_load(vec2 - 1);
index4 = mm_set_epi32(2);
val4 = mm_i32gather(val, index4);
sum4 = mm_fmadd(val4, vec4_x_l, sum4);
m_real vec4 = mm_load(vec2);
index4 = mm_set_epi32(3);
val4 = mm_i32gather(val, index4);
sum4 = mm_fmadd(val4, vec4, sum4);
m_real vec4_x_r = mm_load(vec2 + 1);
index4 = mm_set_epi32(4);
val4 = mm_i32gather(val, index4);
sum4 = mm_fmadd(val4, vec4_x_r, sum4);
m_real vec4_y_r = mm_load(vec2 + nx);
index4 = mm_set_epi32(5);
val4 = mm_i32gather(val, index4);
sum4 = mm_fmadd(val4, vec4_y_r, sum4);
m_real vec4_z_r = mm_load(vec2 + nx*ny);
index4 = mm_set_epi32(6);
val4 = mm_i32gather(val, index4);
sum4 = mm_fmadd(val4, vec4_z_r, sum4);
mm_stream(result, sum4);
val += LENVEC * NR;   result += LENVEC;   vec2 += LENVEC;
} 
val += 1 * NR;   result += 1;   vec2 += 1;
} 
val += nx * NR;   result += nx;   vec2 += nx;
} 
}
inline void multMV_AVX_optimize(TYPE* result, SpMatrix mat, TYPE* vec, int nx, int ny, int nz, TYPE* coeff) {
TYPE * val = mat.value;
TYPE *vec2 = vec;
m_real val41 = mm_set1(coeff[0]);
m_real val42 = mm_set1(coeff[1]);
m_real val43 = mm_set1(coeff[2]);
m_real val44 = mm_set1(coeff[3]);
copyingBorders(vec, nx, ny, nz);
val += nx*ny * NR;   result += nx*ny;   vec2 += nx*ny;
#pragma omp parallel for if (ENABLE_PARALLEL)
for (int z = 1; z < nz - 1; z++) {
val += nx * NR;   result += nx;   vec2 += nx;
for (int y = 1; y < ny - 1; y++) {
val += 1 * NR;   result += 1;   vec2 += 1;
for (int x = 1; x < nx - 1; x += LENVEC) {
m_real vec4_z_l = mm_load(vec2 - nx*ny);
m_real sum4 = _mm256_mul_pd(val44, vec4_z_l);
m_real vec4_y_l = mm_load(vec2 - nx);
sum4 = mm_fmadd(val43, vec4_y_l, sum4);
m_real vec4_x_l = mm_load(vec2 - 1);
sum4 = mm_fmadd(val42, vec4_x_l, sum4);
m_real vec4 = mm_load(vec2);
sum4 = mm_fmadd(val41, vec4, sum4);
m_real vec4_x_r = mm_load(vec2 + 1);
sum4 = mm_fmadd(val42, vec4_x_r, sum4);
m_real vec4_y_r = mm_load(vec2 + nx);
sum4 = mm_fmadd(val43, vec4_y_r, sum4);
m_real vec4_z_r = mm_load(vec2 + nx*ny);
sum4 = mm_fmadd(val44, vec4_z_r, sum4);
mm_stream(result, sum4);
val += LENVEC * NR;   result += LENVEC;   vec2 += LENVEC;
} 
val += 1 * NR;   result += 1;   vec2 += 1;
} 
val += nx * NR;   result += nx;   vec2 += nx;
} 
}
#endif
# define multMV_mklf(result, mat, vec)  mkl_cspblas_scsrgemv("N", &mat.nRows, mat.value, mat.rowIndex, mat.col, vec, result);
# define multMV_mkld(result, mat, vec) mkl_cspblas_dcsrgemv("N", &mat.nRows, mat.value, mat.rowIndex, mat.col, vec, result);
#if FPGA_RUN || CPU_CL_RUN || GPU_CL_RUN
void multMV_altera(TYPE* result, SpMatrix mat, TYPE* vec, int sizeTime ) {
cl_context context = 0;
cl_command_queue commandQueue = 0;
cl_device_id device = 0;
cl_kernel kernel = 0;
cl_program program = 0;
context = createContext();
commandQueue = createCommandQueue(context, &device);
program = createProgram(context, device);
kernel = clCreateKernel(program, "csr_mult_d", NULL);
cl_int err;
cl_mem memResult = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(TYPE)*mat.nRows, NULL, &err);
checkError(err,"memResult");
cl_mem memVec = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(TYPE)*mat.nRows, vec, &err);
checkError(err,"memVec");
cl_mem memCols = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int)*mat.nz, mat.col, &err);
checkError(err,"memCols");
cl_mem memValue = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(TYPE)*mat.nz, mat.value, &err);
checkError(err,"memValue");
cl_mem memRowIndex = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int)*(mat.nRows + 1), mat.rowIndex, &err);
checkError(err,"memRowIndex");
err = clSetKernelArg(kernel, 0, sizeof(int), &mat.nRows);
err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &memRowIndex);
err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &memCols);
err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &memValue);
err |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &memVec);
err |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &memResult);
checkError(err,"clSetKernelArg");
size_t globalWorkSize[1] = {mat.nRows};
size_t max_wg_size, num_wg_sizes = 0;
err = clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), (void *) &max_wg_size, NULL);
checkError(err, "ERROR: clGetKernelWorkGroupInfo failed");
fprintf(stdout, "CL_KERNEL_WORK_GROUP_SIZE = %lu\n", max_wg_size);
cl_ulong a;
err = clGetKernelWorkGroupInfo(kernel, device,  CL_KERNEL_LOCAL_MEM_SIZE, sizeof(cl_ulong), (void *) &a, NULL);
checkError(err, "ERROR: clGetKernelWorkGroupInfo failed");
fprintf(stdout, "CL_KERNEL_LOCAL_MEM_SIZE = %lu\n", a);
size_t attributed[3];
err = clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_COMPILE_WORK_GROUP_SIZE, sizeof(attributed), (void *)attributed, NULL);
checkError(err, "ERROR: clGetKernelWorkGroupInfo failed");
fprintf(stdout, "CL_KERNEL_COMPILE_WORK_GROUP_SIZE  = (%lu, %lu, %lu)\n", attributed[0], attributed[1], attributed[2]);
size_t *localWorkSize2 = default_wg_sizes(&num_wg_sizes,max_wg_size, globalWorkSize);
cl_mem tmp;
for (int i = 0; i <= sizeTime; i++) {
err = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, globalWorkSize, localWorkSize2, 0, NULL, NULL);
checkError(err,  "clEnqueueNDRangeKernel");
tmp = memVec;
memVec = memResult;
memResult = tmp;
clSetKernelArg(kernel, 4, sizeof(cl_mem), &memVec);
clSetKernelArg(kernel, 5, sizeof(cl_mem), &memResult);
}
clFinish(commandQueue);
err = clEnqueueReadBuffer(commandQueue, memVec, CL_TRUE, 0, sizeof(TYPE)*mat.nRows, result, 0, NULL, NULL);
checkError(err,  "clEnqueueReadBuffer: out");
clFinish(commandQueue);
clReleaseMemObject(memResult);
clReleaseMemObject(memVec);
clReleaseMemObject(memCols);
clReleaseMemObject(memValue);
clReleaseMemObject(memRowIndex);
clReleaseKernel(kernel);
clReleaseProgram(program);
clReleaseCommandQueue(commandQueue);
clReleaseContext(context);
}
void naive_formula(TYPE* result, TYPE* vec, const TYPE* const coeff, const int nx, const int ny, const int nz, const int sizeTime) {
const int dims = nx*ny*nz;
cl_device_id device = 0;
cl_context context = createContext();
cl_command_queue commandQueue = createCommandQueue(context, &device);
cl_program program = createProgram(context, device);
cl_kernel kernel = clCreateKernel(program, "naive_formula_d", NULL);
cl_int err;
cl_mem memResult = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(TYPE)*dims, NULL, &err);
checkError(err,"memResult");
cl_mem memVec = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(TYPE)*dims, vec, &err);
checkError(err,"medmVec");
err = clSetKernelArg(kernel,  0, sizeof(int), &nx);
err |= clSetKernelArg(kernel, 1, sizeof(int), &ny);
err |= clSetKernelArg(kernel, 2, sizeof(int), &nz);
err |= clSetKernelArg(kernel, 3, sizeof(TYPE), &coeff[0]);
err |= clSetKernelArg(kernel, 4, sizeof(TYPE), &coeff[1]);
err |= clSetKernelArg(kernel, 5, sizeof(TYPE), &coeff[2]);
err |= clSetKernelArg(kernel, 6, sizeof(TYPE), &coeff[3]);
err |= clSetKernelArg(kernel, 7, sizeof(cl_mem), &memVec);
err |= clSetKernelArg(kernel, 8, sizeof(cl_mem), &memResult);
checkError(err,"clSetKernelArg");
size_t globalWorkSize[] = { nx, ny, nz };
size_t attributed[3];
err = clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_COMPILE_WORK_GROUP_SIZE, sizeof(attributed), (void *)attributed, NULL);
checkError(err, "ERROR: clGetKernelWorkGroupInfo failed");
fprintf(stdout, "CL_KERNEL_COMPILE_WORK_GROUP_SIZE  = (%lu, %lu, %lu)\n", attributed[0], attributed[1], attributed[2]);
cl_mem tmp;
for (int i = 0; i < sizeTime; i++) {
err = clEnqueueNDRangeKernel(commandQueue, kernel, 3, NULL, globalWorkSize, NULL, 0, NULL, NULL);
checkError(err,  "clEnqueueNDRangeKernel");
tmp = memVec;
memVec = memResult;
memResult = tmp;
clSetKernelArg(kernel, 7, sizeof(cl_mem), &memVec);
clSetKernelArg(kernel, 8, sizeof(cl_mem), &memResult);
}
clFinish(commandQueue);
err = clEnqueueReadBuffer(commandQueue, memVec, CL_TRUE, 0, sizeof(TYPE)*dims, result, 0, NULL, NULL);
checkError(err,  "clEnqueueReadBuffer: out");
clFinish(commandQueue);
clReleaseMemObject(memResult);
clReleaseMemObject(memVec);
clReleaseKernel(kernel);
clReleaseProgram(program);
clReleaseCommandQueue(commandQueue);
clReleaseContext(context);
}
#endif
void multMV(TYPE *result, SpMatrix mat, TYPE *vec, int nx, int ny, int nz, TYPE *coeff) {
#if MKL_RUN
#if defined(DOUBLE_TYPE)
multMV_mkld(result, mat, vec);
#elif defined(FLOAT_TYPE)
multMV_mklf(result, mat, vec);
#endif
#elif AVX2_RUN
multMV_AVX_optimize(result,mat, vec, nx, ny, nz, coeff);
#else
multMV_default(result, mat, vec);
#endif
}
void sumV(TYPE *result, const TYPE *const U, const TYPE *const k1,
const TYPE *const k2, const TYPE *const k3, const TYPE *const k4, const int N, const TYPE h) {
#pragma omp parallel for if (ENABLE_PARALLEL)
for (int i = 0; i < N; i++) {
result[i] = U[i] + h * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
}
}
void printSpMat(SpMatrix mat) {
for (int i = 0; i < mat.nRows; i++) {
for (int j = 0; j < mat.nRows; j++)
printf("%.0lf", procedure(mat, i, j));
printf("\n");
}
}
TYPE procedure(SpMatrix mat, int i, int j) {
TYPE result = 0;
int N1 = mat.rowIndex[i];
int N2 = mat.rowIndex[i + 1];
for (int k = N1; k < N2; k++) {
if (mat.col[k] == j) {
result = mat.value[k];
break;
}
}
return result;
}
void denseMult(TYPE **result, TYPE **mat, TYPE *vec, int dim) {
memset(*result, 0, dim * sizeof(TYPE));
for (int x = 0; x < dim; x++) {
for (int i = 0; i < dim; i++)
(*result)[x] += mat[x][i] * vec[i];
}
}
