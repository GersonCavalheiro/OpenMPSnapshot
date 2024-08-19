#include <stdio.h>
#include <stdlib.h>
#include <arm_neon.h> 
#include <omp.h>
#include <time.h>
#define SIZE_M 10000000
float x[SIZE_M], y[SIZE_M];
float32_t x_par[SIZE_M], y_par[SIZE_M];
float a = 0;
float32x4_t a_par;
int main (){
a_par = vdupq_n_f32(0);
double start_time, run_time;
srand(time(NULL));
int procs = omp_get_num_procs();
omp_set_num_threads(procs);
#pragma omp parallel  
{
#pragma omp for
for (int i = 0; i < SIZE_M; ++i)
{
x[i] = x_par[i] = rand()%100;
y[i] = y_par[i] = rand()%100;
}
}
start_time = omp_get_wtime();
float element = 0;
for (int i = 0; i < SIZE_M; ++i)
{
element = x[i]*y[i];
a += element;
}
run_time = omp_get_wtime() - start_time;
printf("SERIAL: Completed in %f seconds\n", run_time);
start_time = omp_get_wtime();
#pragma omp parallel  
{
float32x4_t element_par = vdupq_n_f32(0);
#pragma omp for private(element_par)
for (int i = 0; i < SIZE_M; i+=4)
{
float32x4_t y_vec = vld1q_f32(y_par+i); 
float32x4_t x_vec = vld1q_f32(x_par+i); 
element_par = vmulq_f32(x_vec, y_vec);
a_par = vaddq_f32(element_par, a_par);
}
}
a = a_par[0] + a_par[1] + a_par[2] + a_par[3];
run_time = omp_get_wtime() - start_time;
printf("PARALLEL: Completed in %f seconds\n", run_time);
return 0;
}