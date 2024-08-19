#include <stdio.h>
#ifdef _OPENMP
#include <omp.h>
#endif


void mxv(int m, int n, double *a, double *b, double *c) {
int i, j;
double sum;


for (i=0; i<m; i++) {
sum = 0.0;
for (j=0; j<n; j++) {
sum += b[i*n+j] * c[j];
}
a[i] = sum;
}
}

int main(int argc, char *argv[]) {

}
