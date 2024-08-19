#include <stdio.h>
void do_openmp_test() {
#pragma omp parallel for
for (int i = 0; i < 10; i++) {
printf("%d\n", i);
}
}
