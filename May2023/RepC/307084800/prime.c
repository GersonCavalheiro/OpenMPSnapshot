#include <omp.h>
#include <stdio.h>
#include <math.h>
int main(int argc, char **argv) {
int n = atoi(argv[1]);
int primes[n+1];
#pragma omp parallel
#pragma omp for
for (int i = 0; i <= n; i++) {
primes[i] = 0;
}
int i = 2;
int lim = sqrt(n);
#pragma omp for
for (i = 2; i <= lim; i++) { 
for (int j = i * i; j <= n; j += i) {
primes[j] = 1; 
}
}
#pragma omp for
for (int i = 2; i <= n; i++) {
if (primes[i] == 0) {
printf("%d\n", i);
}
} 
}