#include <stdio.h>
#include <stdlib.h>
int main(int argc, char *argv[]) {
int *A; 
int N = 100;
A = (int*) malloc(sizeof(int) * N);
#pragma omp parallel for shared(A)
for(int i = 0; i < N; i++) {
A[i] = i;
if (i == 1) 
{ 
A[0] = 1; 
}
}
free(A);
return 0;
}
