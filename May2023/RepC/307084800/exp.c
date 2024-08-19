#include <stdio.h>
#include <omp.h>
double calc(uint32_t x_last, uint32_t num_threads)
{
double* loc_res = (double*)calloc(num_threads, sizeof(double));
double* loc_fact = (double*)malloc(num_threads * sizeof(double));
#pragma omp parallel num_threads(num_threads) 
{
int tid = omp_get_thread_num();
loc_fact[tid] = 1.0;
#pragma omp for 
for (uint32_t i = 1; i < x_last; i++) {
loc_fact[tid] = loc_fact[tid]/(double)i;
loc_res[tid] += loc_fact[tid];
} 
}
double res = 1.0;
double fact = 1.0;
for (uint32_t tid = 0; tid < num_threads; tid++) {
res += loc_res[tid] * fact;
fact *= loc_fact[tid];
}
free(loc_fact);
free(loc_res);
return res;
}
int main(int argc, char** argv)
{
uint32_t x_last = atoi(argv[1]);
uint32_t num_threads = atoi(argv[2]);
double res = calc(x_last, num_threads);
printf("%lf\n", res);
return 0;
}
