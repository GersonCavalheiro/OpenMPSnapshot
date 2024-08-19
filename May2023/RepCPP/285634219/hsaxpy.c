

#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "wtcalc.h"
#include "hsaxpy.h"

void hsaxpy(const int n,
const float a,
const float *x,
float *y)
{
struct timespec rt[2];


clock_gettime(CLOCK_REALTIME, rt + 0);
#pragma omp parallel for simd schedule(simd:static) \
default(none) shared(a, n, x, y)
for (int i = 0; i < n; i++) {
y[i] = a * x[i] + y[i];
}
clock_gettime(CLOCK_REALTIME, rt + 1);
if (wtcalc >= 0.0) {
wtcalc += (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
}
}
