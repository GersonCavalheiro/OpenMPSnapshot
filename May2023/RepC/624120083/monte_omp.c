#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "monte.h"
int main(int argc, char **argv) {
if (argc != 2) {
fprintf(stderr, "Usage: %s n_pts\n", argv[0]);
return EXIT_FAILURE;
}
int n_pts = atoi(argv[1]);
int count = 0;
#pragma omp parallel for schedule(dynamic) shared(count)
for (int i = 0; i < n_pts; ++i) {
point_t *rpt = rand_pt();
if (f(rpt)) {
#pragma omp atomic
count++;
}
}
double pi = count * 4. / n_pts;
printf("PI = %f\n", pi);
return EXIT_SUCCESS;
}
