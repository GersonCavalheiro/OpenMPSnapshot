#include <omp.h>
#include <iostream>
#include <math.h>
#include <stdlib.h>
int main (){
int N = 1500;
double *A, *B, *C;
double start_time, run_time;
posix_memalign((void**)&A, 64, N*N*sizeof(double));
posix_memalign((void**)&B, 64, N*N*sizeof(double));
posix_memalign((void**)&C, 64, N*N*sizeof(double));
omp_set_num_threads(8);
start_time = omp_get_wtime();
#pragma omp parallel for
for(int i=0; i<N; i++){
for(int k=0; k<N; k++){
for(int j=0; j<N; j++){
C[i*N+j] += A[i*N+k]*B[k*N+j];
}
}
}
run_time = omp_get_wtime() - start_time;
std::cout << "Tiempo: " << run_time << " seconds" << std::endl;
}
