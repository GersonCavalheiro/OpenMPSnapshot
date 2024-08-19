#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <omp.h>

bool isPrime(int n) {
if (n == 2) {
return true;
}
int threshold = sqrt(n) + 1;
for (int i = 2; i < threshold; i++) {
if (n % i == 0) {
return false;
}
}
return true;
}

int main(int argc, char* argv[]) {
int a = atoi(argv[1]);
int b = atoi(argv[2]);

int count = 0;

#pragma omp parallel for reduction(+: count)
for (int i = a; i <= b; i++) {
if (isPrime(i)) {
count += 1;
}
}

printf("there are %d primes between %d and %d\n", count, a, b);
}
