





#include "openacc.h"

void openACC( int const    n,
float const  alpha,
float const *A,
float const *B,
float const  beta,
float *      C,
int const &  loops ) {

for ( int l = 0; l < loops; l++ ) {

#pragma acc kernels copyin( A [0:( n * n )], B [0:( n * n )] ) copyout( C [0:( n * n )] )
#pragma acc loop independent
for ( int i = 0; i < n; ++i ) {
#pragma acc loop independent
for ( int j = 0; j < n; ++j ) {
float prod = 0.0f;
#pragma acc loop independent reduction( + : prod )
for ( int k = 0; k < n; ++k ) {
prod += A[k * n + i] * B[j * n + k];
}  
C[j * n + i] = alpha * prod + beta * C[j * n + i];
}  
}      
}          
}  
