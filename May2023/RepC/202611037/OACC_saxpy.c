#include <stdlib.h>
#include <omp.h>
void saxpy(int n, float a, float *x, float *restrict y){
#pragma acc kernels
for (int i = 0; i < n; ++i)
y[i] = a * x[i] + y[i];
}
int main(int argc, char **argv){
int N = 1<<20; 
double t_inicial, t_final;
if (argc > 1)
N = atoi(argv[1]);
float *x = (float*)malloc(N * sizeof(float));
float *y = (float*)malloc(N * sizeof(float));
t_inicial = omp_get_wtime();
for (int i = 0; i < N; ++i) {
x[i] = 2.0f;
y[i] = 1.0f;
}
saxpy(N, 3.0f, x, y);
t_final = omp_get_wtime();
printf("tempo: %3f\n", t_final-t_inicial);
return 0;
}
