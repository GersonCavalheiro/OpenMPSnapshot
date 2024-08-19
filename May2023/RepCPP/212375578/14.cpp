#include <cstdio>
#include <omp.h>

int main() {
int n = 210;
int sqr = 0;

int b = n * 2;
#pragma omp parallel for reduction(+: sqr)
for (int i = 1; i < b; i += 2) {
sqr += i;
}

printf("square of %d = %d\n", n, sqr);
}
