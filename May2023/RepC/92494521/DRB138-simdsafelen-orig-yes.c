#include <stdio.h>
#include <omp.h>
int main(){
int i, m=1, n=4;
int b[4] = {};
#pragma omp simd safelen(2)
for (i = m; i<n; i++)
b[i] = b[i-m] - 1.0f;
printf("Expected: -1; Real: %d\n",b[3]);
return 0;
}
