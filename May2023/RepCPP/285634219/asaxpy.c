

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <CL/sycl.hpp>
#include "oneapi/mkl/blas.hpp"
#include "wtcalc.h"
#include "asaxpy.h"

void asaxpy(const int n,
const float a,
const float *x,
float *y,
const int ial)
{
#ifdef USE_GPU
sycl::gpu_selector dev_sel;
#else
sycl::cpu_selector dev_sel;
#endif
sycl::queue q(dev_sel);

struct timespec rt[2];
int m = (n >> 4);

switch (ial) {
case 0:

#pragma omp target data  device(0) map(to:a, n, x[0:n]) map(tofrom:y[0:n])
{
clock_gettime(CLOCK_REALTIME, rt + 0);
#pragma omp target teams distribute parallel for device(0) \
num_teams(128) num_threads(128) dist_schedule(static, 128) shared(a, n, x, y)
for (int i = 0; i < n; ++i) {
y[i] = a * x[i] + y[i];
}
clock_gettime(CLOCK_REALTIME, rt + 1);
}
break;
case 1:

#pragma omp target data  device(0) \
map(to:a, n, x[0:n]) map(tofrom:y[0:n])
{
clock_gettime(CLOCK_REALTIME, rt + 0);
#pragma omp target teams distribute parallel for device(0) \
num_teams(65536) num_threads(1024) dist_schedule(static, 1024) shared(a, n, x, y)
for (int i = 0; i < n; ++i) {
y[i] = a * x[i] + y[i];
}
clock_gettime(CLOCK_REALTIME, rt + 1);
}
break;
case 2:

#pragma omp target data  device(0) \
map(to:a, m, x[0:n]) map(tofrom:y[0:n])
{
clock_gettime(CLOCK_REALTIME, rt + 0);
#pragma omp target teams distribute parallel for device(0) \
num_teams(65536/2) num_threads(128) dist_schedule(static, 128) shared(a, m, x, y)
for (int i = 0; i < m; ++i) {
y[i          ] = a * x[i          ] + y[i          ];
y[i +       m] = a * x[i +       m] + y[i +       m];
y[i + 0x2 * m] = a * x[i + 0x2 * m] + y[i + 0x2 * m];
y[i + 0x3 * m] = a * x[i + 0x3 * m] + y[i + 0x3 * m];
y[i + 0x4 * m] = a * x[i + 0x4 * m] + y[i + 0x4 * m];
y[i + 0x5 * m] = a * x[i + 0x5 * m] + y[i + 0x5 * m];
y[i + 0x6 * m] = a * x[i + 0x6 * m] + y[i + 0x6 * m];
y[i + 0x7 * m] = a * x[i + 0x7 * m] + y[i + 0x7 * m];
y[i + 0x8 * m] = a * x[i + 0x8 * m] + y[i + 0x8 * m];
y[i + 0x9 * m] = a * x[i + 0x9 * m] + y[i + 0x9 * m];
y[i + 0xa * m] = a * x[i + 0xa * m] + y[i + 0xa * m];
y[i + 0xb * m] = a * x[i + 0xb * m] + y[i + 0xb * m];
y[i + 0xc * m] = a * x[i + 0xc * m] + y[i + 0xc * m];
y[i + 0xd * m] = a * x[i + 0xd * m] + y[i + 0xd * m];
y[i + 0xe * m] = a * x[i + 0xe * m] + y[i + 0xe * m];
y[i + 0xf * m] = a * x[i + 0xf * m] + y[i + 0xf * m];
}
clock_gettime(CLOCK_REALTIME, rt + 1);
}
break;
case 3:

#pragma omp target data  device(0) \
map(to:a, m, x[0:n]) map(tofrom:y[0:n])
{
clock_gettime(CLOCK_REALTIME, rt + 0);
#pragma omp target teams distribute parallel for device(0) \
num_teams(4096) num_threads(128) dist_schedule(static, 128) shared(a, m, x, y)
for (int i = 0; i < m; ++i) {
y[i          ] = a * x[i          ] + y[i          ];
y[i +       m] = a * x[i +       m] + y[i +       m];
y[i + 0x2 * m] = a * x[i + 0x2 * m] + y[i + 0x2 * m];
y[i + 0x3 * m] = a * x[i + 0x3 * m] + y[i + 0x3 * m];
y[i + 0x4 * m] = a * x[i + 0x4 * m] + y[i + 0x4 * m];
y[i + 0x5 * m] = a * x[i + 0x5 * m] + y[i + 0x5 * m];
y[i + 0x6 * m] = a * x[i + 0x6 * m] + y[i + 0x6 * m];
y[i + 0x7 * m] = a * x[i + 0x7 * m] + y[i + 0x7 * m];
y[i + 0x8 * m] = a * x[i + 0x8 * m] + y[i + 0x8 * m];
y[i + 0x9 * m] = a * x[i + 0x9 * m] + y[i + 0x9 * m];
y[i + 0xa * m] = a * x[i + 0xa * m] + y[i + 0xa * m];
y[i + 0xb * m] = a * x[i + 0xb * m] + y[i + 0xb * m];
y[i + 0xc * m] = a * x[i + 0xc * m] + y[i + 0xc * m];
y[i + 0xd * m] = a * x[i + 0xd * m] + y[i + 0xd * m];
y[i + 0xe * m] = a * x[i + 0xe * m] + y[i + 0xe * m];
y[i + 0xf * m] = a * x[i + 0xf * m] + y[i + 0xf * m];
}
clock_gettime(CLOCK_REALTIME, rt + 1);
}
break;
case 4:

#pragma omp target data  device(0) \
map(to:a, x[0:n]) map(tofrom:y[0:n])
{
clock_gettime(CLOCK_REALTIME, rt + 0);
#pragma omp target teams distribute parallel for device(0) \
num_teams(65536) thread_limit(512) dist_schedule(static, 512) \
collapse(2) shared(a, x, y)
for (int j = 0; j < 65536; ++j) {
for (int i = 0; i < 512; ++i) { 
y[j * 1024 + i      ] += a * x[j * 1024 + i      ];
y[j * 1024 + i + 512] += a * x[j * 1024 + i + 512];
}
}
clock_gettime(CLOCK_REALTIME, rt + 1);
}
break;

default:


sycl::buffer<float, 1> x_dev(x, n);
sycl::buffer<float, 1> y_dev(y, n);
clock_gettime(CLOCK_REALTIME, rt + 0);
try {
oneapi::mkl::blas::axpy(q, n, a, x_dev, 1, y_dev, 1);
q.wait();
}
catch(sycl::exception const& e) {
std::cout << "\t\tCaught synchronous SYCL exception during AXPY:\n"
<< e.what() << std::endl;
}
clock_gettime(CLOCK_REALTIME, rt + 1);
break;
} 

if (wtcalc >= 0.0) {
wtcalc += (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
}
}
