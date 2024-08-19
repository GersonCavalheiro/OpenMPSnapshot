#include <stdlib.h>
#include <stdio.h>
#include <stdlib.h>
int main(int argc, char** argv){
float A[1000];
#pragma acc data copy(A)
{
#pragma acc kernels
for( int iter = 1; iter < 1000 ; iter++){
A[iter] = 1.0;
}
A[10] = 2.0;
}
printf("A[10] = %f\n", A[10]);
}