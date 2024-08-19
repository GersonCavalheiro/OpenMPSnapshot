





#include <omp.h>

#include "openmp.h"
#include "timer.h"

void openMP( int const &  n,
float const &alpha,
float const *A,
float const *B,
float const &beta,
float *      C,
int const &  loops ) {

Timer timer {};

omp_set_num_threads( omp_get_max_threads( ) );

timer.startCPUTimer( );

for ( int l = 0; l < loops; l++ ) {

int i, j, k;

#pragma omp parallel for shared( A, B, C, n ) private( i, j, k ) schedule( static )
for ( i = 0; i < n; ++i ) {
for ( j = 0; j < n; ++j ) {
float prod = 0.0f;
for ( k = 0; k < n; ++k ) {
prod += A[k * n + i] * B[j * n + k];
}  
C[j * n + i] = alpha * prod + beta * C[j * n + i];
}  
}      
}          

timer.stopAndPrintCPU( loops );

}  
