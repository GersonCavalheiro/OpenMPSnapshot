#include <omp.h>
#include <stdio.h>
#include <limits.h>
#define N 1000
#define CHUNKSIZE 100
int main() {
int a[N], b[N], c[N], chunk = CHUNKSIZE;
int i, j, k;
#pragma omp parallel shared(a, b, c, chunk) private(i, j, k)
{
for(i = 0; i < N; i++) {
a[i] = i;
}
for(j = 0; j < N; j++) {
b[j] = N-j;
}
for(k = 0; k < N; k++) {
c[k] = b[k] + a[k];
}
}
return 0;
}
