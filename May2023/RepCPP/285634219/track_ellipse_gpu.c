#include "track_ellipse.h"


#define LOCAL_WORK_SIZE 256
#define FP_TYPE float
#define FP_CONST(num) num##f
#define PI_FP32 FP_CONST(3.14159)
#define ONE_OVER_PI (FP_CONST(1.0) / PI_FP32)
#define MU FP_CONST(0.5)
#define LAMBDA (FP_CONST(8.0) * MU + FP_CONST(1.0))
#define NEXT_LOWEST_POWER_OF_TWO 256

#pragma omp declare target
FP_TYPE heaviside(FP_TYPE x) {
return (atanf(x) * ONE_OVER_PI) + FP_CONST(0.5);
}
#pragma omp end declare target

void IMGVF_GPU(MAT **IE, MAT **IMGVF, 
double vx, double vy, double e, int max_iterations, double cutoff, int num_cells) {

size_t mem_size = sizeof(int) * num_cells;
int* host_I_offsets = (int *) malloc(mem_size);

int* host_m_array = (int *) malloc(mem_size);
int* host_n_array = (int *) malloc(mem_size);

int i, j;
size_t total_size = 0;
for (int cell_num = 0; cell_num < num_cells; cell_num++) {
MAT *I = IE[cell_num];
size_t size = I->m * I->n;
total_size += size;
}
size_t total_mem_size = total_size * sizeof(float);

float* host_I_all = (float *) malloc(total_mem_size);
float* host_IMGVF_all = (float *) malloc(total_mem_size);

int offset = 0;
for (int cell_num = 0; cell_num < num_cells; cell_num++) {
MAT *I = IE[cell_num];

int m = I->m, n = I->n;
int size = m * n;

host_m_array[cell_num] = m;
host_n_array[cell_num] = n;

host_I_offsets[cell_num] = offset;

for (i = 0; i < m; i++)
for (j = 0; j < n; j++)
host_I_all[offset + (i * n) + j] = host_IMGVF_all[offset + (i * n) + j] = 
(float) m_get_val(I, i, j);

offset += size;
}



float vx_float = (float) vx;
float vy_float = (float) vy;
float e_float = (float) e;
float cutoff_float = (float) cutoff;

#pragma omp target data map(to: host_I_offsets[0:num_cells],\
host_m_array[0:num_cells], \
host_n_array[0:num_cells], \
host_I_all[0:total_size]) \
map(tofrom: host_IMGVF_all[0:total_size])
{
auto start = std::chrono::steady_clock::now();

#include "kernel_IMGVF.h"

auto end = std::chrono::steady_clock::now();
auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
printf("Kernel execution time (IMGVF): %f (s)\n", time * 1e-9f);
}

offset = 0;  
for (int cell_num = 0; cell_num < num_cells; cell_num++) {
MAT *IMGVF_out = IMGVF[cell_num];

int m = IMGVF_out->m, n = IMGVF_out->n, i, j;
for (i = 0; i < m; i++)
for (j = 0; j < n; j++) {
#ifdef DEBUG
printf("host_IMGVF: %f\n",host_IMGVF_all[offset + (i * n) + j]);
#endif

m_set_val(IMGVF_out, i, j, (double) host_IMGVF_all[offset + (i * n) + j]);
}

offset += (m * n);
}

free(host_m_array);
free(host_n_array);
free(host_I_all);
free(host_I_offsets);
free(host_IMGVF_all);
}
