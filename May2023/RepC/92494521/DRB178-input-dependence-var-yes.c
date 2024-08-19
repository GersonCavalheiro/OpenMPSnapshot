#include <stdio.h>
#include <stdlib.h>
void load_from_input(int *data, int size)
{
for(int i = 0; i < size; i++) {
data[i] = size-i;
} 
}
int main(int argc, char *argv[]) {
int *A; 
int N = 100;
if (argc>1)
N = atoi(argv[1]);
A = (int*) malloc(sizeof(int) * N);
load_from_input(A, N);
#pragma omp parallel for shared(A)
for(int i = 0; i < N; i++) {
A[i] = i;
if (N > 10000) 
{ 
A[0] = 1; 
}
}
free(A);
return 0;
}
