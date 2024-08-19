#include <omp.h>
#include <stdio.h>
#define n 100000
int main(int argc, char **argv) {
int a[n];
int b[n];
for (int i = 0; i < n; i++) {
a[i] = i;
}
b[0] = 0;
b[n - 1] = a[n - 1]; 
#pragma omp parallel 
{	
#pragma omp for
for (int i = 1; i < n - 1; i++) {
b[i] = a[i - 1] * a[i] * a[i + 1]/3;
}
}
int l = n > 10 ? 10 : n;
for (int i = 0; i < l; i++) {
printf("%d ", b[i]);
}
}