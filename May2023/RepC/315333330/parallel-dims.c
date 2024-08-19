#include <limits.h>
#include <openacc.h>
#pragma acc routine seq
static unsigned int __attribute__ ((optimize ("O2"))) acc_gang ()
{
if (acc_on_device ((int) acc_device_host))
return 0;
else if (acc_on_device ((int) acc_device_nvidia))
{
unsigned int r;
asm volatile ("mov.u32 %0,%%ctaid.x;" : "=r" (r));
return r;
}
else
__builtin_abort ();
}
#pragma acc routine seq
static unsigned int __attribute__ ((optimize ("O2"))) acc_worker ()
{
if (acc_on_device ((int) acc_device_host))
return 0;
else if (acc_on_device ((int) acc_device_nvidia))
{
unsigned int r;
asm volatile ("mov.u32 %0,%%tid.y;" : "=r" (r));
return r;
}
else
__builtin_abort ();
}
#pragma acc routine seq
static unsigned int __attribute__ ((optimize ("O2"))) acc_vector ()
{
if (acc_on_device ((int) acc_device_host))
return 0;
else if (acc_on_device ((int) acc_device_nvidia))
{
unsigned int r;
asm volatile ("mov.u32 %0,%%tid.x;" : "=r" (r));
return r;
}
else
__builtin_abort ();
}
int main ()
{
acc_init (acc_device_default);
{
#define GANGS 0 
int gangs_actual = GANGS;
int gangs_min, gangs_max, workers_min, workers_max, vectors_min, vectors_max;
gangs_min = workers_min = vectors_min = INT_MAX;
gangs_max = workers_max = vectors_max = INT_MIN;
#pragma acc parallel copy (gangs_actual) reduction (min: gangs_min, workers_min, vectors_min) reduction (max: gangs_max, workers_max, vectors_max) num_gangs (GANGS) 
{
gangs_actual = 1;
for (int i = 100 * gangs_actual; i > -100 * gangs_actual; --i)
{
#if 0
gangs_min = gangs_max = acc_gang ();
workers_min = workers_max = acc_worker ();
vectors_min = vectors_max = acc_vector ();
#else
int gangs = acc_gang ();
gangs_min = (gangs_min < gangs) ? gangs_min : gangs;
gangs_max = (gangs_max > gangs) ? gangs_max : gangs;
int workers = acc_worker ();
workers_min = (workers_min < workers) ? workers_min : workers;
workers_max = (workers_max > workers) ? workers_max : workers;
int vectors = acc_vector ();
vectors_min = (vectors_min < vectors) ? vectors_min : vectors;
vectors_max = (vectors_max > vectors) ? vectors_max : vectors;
#endif
}
}
if (gangs_actual != 1)
__builtin_abort ();
if (gangs_min != 0 || gangs_max != gangs_actual - 1
|| workers_min != 0 || workers_max != 0
|| vectors_min != 0 || vectors_max != 0)
__builtin_abort ();
#undef GANGS
}
{
#define GANGS 0 
int gangs_actual = GANGS;
int gangs_min, gangs_max, workers_min, workers_max, vectors_min, vectors_max;
gangs_min = workers_min = vectors_min = INT_MAX;
gangs_max = workers_max = vectors_max = INT_MIN;
#pragma acc parallel copy (gangs_actual) num_gangs (GANGS) 
{
gangs_actual = 1;
#pragma acc loop gang reduction (min: gangs_min, workers_min, vectors_min) reduction (max: gangs_max, workers_max, vectors_max)
for (int i = 100 * gangs_actual; i > -100 * gangs_actual; --i)
{
gangs_min = gangs_max = acc_gang ();
workers_min = workers_max = acc_worker ();
vectors_min = vectors_max = acc_vector ();
}
}
if (gangs_actual != 1)
__builtin_abort ();
if (gangs_min != 0 || gangs_max != gangs_actual - 1
|| workers_min != 0 || workers_max != 0
|| vectors_min != 0 || vectors_max != 0)
__builtin_abort ();
#undef GANGS
}
{
#define WORKERS 0 
int workers_actual = WORKERS;
int gangs_min, gangs_max, workers_min, workers_max, vectors_min, vectors_max;
gangs_min = workers_min = vectors_min = INT_MAX;
gangs_max = workers_max = vectors_max = INT_MIN;
#pragma acc parallel copy (workers_actual) num_workers (WORKERS) 
{
workers_actual = 1;
#pragma acc loop worker reduction (min: gangs_min, workers_min, vectors_min) reduction (max: gangs_max, workers_max, vectors_max)
for (int i = 100 * workers_actual; i > -100 * workers_actual; --i)
{
gangs_min = gangs_max = acc_gang ();
workers_min = workers_max = acc_worker ();
vectors_min = vectors_max = acc_vector ();
}
}
if (workers_actual != 1)
__builtin_abort ();
if (gangs_min != 0 || gangs_max != 0
|| workers_min != 0 || workers_max != workers_actual - 1
|| vectors_min != 0 || vectors_max != 0)
__builtin_abort ();
#undef WORKERS
}
{
#define VECTORS 0 
int vectors_actual = VECTORS;
int gangs_min, gangs_max, workers_min, workers_max, vectors_min, vectors_max;
gangs_min = workers_min = vectors_min = INT_MAX;
gangs_max = workers_max = vectors_max = INT_MIN;
#pragma acc parallel copy (vectors_actual)  vector_length (VECTORS) 
{
if (acc_on_device (acc_device_nvidia))
vectors_actual = 32;
else
vectors_actual = 1;
#pragma acc loop vector reduction (min: gangs_min, workers_min, vectors_min) reduction (max: gangs_max, workers_max, vectors_max)
for (int i = 100 * vectors_actual; i > -100 * vectors_actual; --i)
{
gangs_min = gangs_max = acc_gang ();
workers_min = workers_max = acc_worker ();
vectors_min = vectors_max = acc_vector ();
}
}
if (acc_get_device_type () == acc_device_nvidia)
{
if (vectors_actual != 32)
__builtin_abort ();
}
else
if (vectors_actual != 1)
__builtin_abort ();
if (gangs_min != 0 || gangs_max != 0
|| workers_min != 0 || workers_max != 0
|| vectors_min != 0 || vectors_max != vectors_actual - 1)
__builtin_abort ();
#undef VECTORS
}
{
int gangs = 12345;
int gangs_actual = gangs;
int gangs_min, gangs_max, workers_min, workers_max, vectors_min, vectors_max;
gangs_min = workers_min = vectors_min = INT_MAX;
gangs_max = workers_max = vectors_max = INT_MIN;
#pragma acc parallel copy (gangs_actual) reduction (min: gangs_min, workers_min, vectors_min) reduction (max: gangs_max, workers_max, vectors_max) num_gangs (gangs)
{
if (acc_on_device (acc_device_host))
{
gangs_actual = 1;
}
for (int i = 100 ; i > -100 ; --i)
{
gangs_min = gangs_max = acc_gang ();
workers_min = workers_max = acc_worker ();
vectors_min = vectors_max = acc_vector ();
}
}
if (gangs_actual < 1)
__builtin_abort ();
if (gangs_min != 0 || gangs_max != gangs_actual - 1
|| workers_min != 0 || workers_max != 0
|| vectors_min != 0 || vectors_max != 0)
__builtin_abort ();
}
{
int gangs = 12345;
int gangs_actual = gangs;
int gangs_min, gangs_max, workers_min, workers_max, vectors_min, vectors_max;
gangs_min = workers_min = vectors_min = INT_MAX;
gangs_max = workers_max = vectors_max = INT_MIN;
#pragma acc parallel copy (gangs_actual) num_gangs (gangs)
{
if (acc_on_device (acc_device_host))
{
gangs_actual = 1;
}
#pragma acc loop gang reduction (min: gangs_min, workers_min, vectors_min) reduction (max: gangs_max, workers_max, vectors_max)
for (int i = 100 * gangs_actual; i > -100 * gangs_actual; --i)
{
gangs_min = gangs_max = acc_gang ();
workers_min = workers_max = acc_worker ();
vectors_min = vectors_max = acc_vector ();
}
}
if (gangs_actual < 1)
__builtin_abort ();
if (gangs_min != 0 || gangs_max != gangs_actual - 1
|| workers_min != 0 || workers_max != 0
|| vectors_min != 0 || vectors_max != 0)
__builtin_abort ();
}
{
#define WORKERS 2 << 20
int workers_actual = WORKERS;
int gangs_min, gangs_max, workers_min, workers_max, vectors_min, vectors_max;
gangs_min = workers_min = vectors_min = INT_MAX;
gangs_max = workers_max = vectors_max = INT_MIN;
#pragma acc parallel copy (workers_actual)  num_workers (WORKERS)
{
if (acc_on_device (acc_device_host))
{
workers_actual = 1;
}
else if (acc_on_device (acc_device_nvidia))
{
workers_actual = 32;
}
else
__builtin_abort ();
#pragma acc loop worker reduction (min: gangs_min, workers_min, vectors_min) reduction (max: gangs_max, workers_max, vectors_max)
for (int i = 100 * workers_actual; i > -100 * workers_actual; --i)
{
gangs_min = gangs_max = acc_gang ();
workers_min = workers_max = acc_worker ();
vectors_min = vectors_max = acc_vector ();
}
}
if (workers_actual < 1)
__builtin_abort ();
if (gangs_min != 0 || gangs_max != 0
|| workers_min != 0 || workers_max != workers_actual - 1
|| vectors_min != 0 || vectors_max != 0)
__builtin_abort ();
#undef WORKERS
}
{
int workers = 2 << 20;
if (acc_get_device_type () == acc_device_nvidia)
workers = 32;
int workers_actual = workers;
int gangs_min, gangs_max, workers_min, workers_max, vectors_min, vectors_max;
gangs_min = workers_min = vectors_min = INT_MAX;
gangs_max = workers_max = vectors_max = INT_MIN;
#pragma acc parallel copy (workers_actual) num_workers (workers)
{
if (acc_on_device (acc_device_host))
{
workers_actual = 1;
}
else if (acc_on_device (acc_device_nvidia))
{
}
else
__builtin_abort ();
#pragma acc loop worker reduction (min: gangs_min, workers_min, vectors_min) reduction (max: gangs_max, workers_max, vectors_max)
for (int i = 100 * workers_actual; i > -100 * workers_actual; --i)
{
gangs_min = gangs_max = acc_gang ();
workers_min = workers_max = acc_worker ();
vectors_min = vectors_max = acc_vector ();
}
}
if (workers_actual < 1)
__builtin_abort ();
if (gangs_min != 0 || gangs_max != 0
|| workers_min != 0 || workers_max != workers_actual - 1
|| vectors_min != 0 || vectors_max != 0)
__builtin_abort ();
}
{
#define VECTORS 2 << 20
int vectors_actual = VECTORS;
int gangs_min, gangs_max, workers_min, workers_max, vectors_min, vectors_max;
gangs_min = workers_min = vectors_min = INT_MAX;
gangs_max = workers_max = vectors_max = INT_MIN;
#pragma acc parallel copy (vectors_actual)  vector_length (VECTORS)
{
if (acc_on_device (acc_device_host))
{
vectors_actual = 1;
}
else if (acc_on_device (acc_device_nvidia))
{
vectors_actual = 32;
}
else
__builtin_abort ();
#pragma acc loop vector reduction (min: gangs_min, workers_min, vectors_min) reduction (max: gangs_max, workers_max, vectors_max)
for (int i = 100 * vectors_actual; i > -100 * vectors_actual; --i)
{
gangs_min = gangs_max = acc_gang ();
workers_min = workers_max = acc_worker ();
vectors_min = vectors_max = acc_vector ();
}
}
if (vectors_actual < 1)
__builtin_abort ();
if (gangs_min != 0 || gangs_max != 0
|| workers_min != 0 || workers_max != 0
|| vectors_min != 0 || vectors_max != vectors_actual - 1)
__builtin_abort ();
#undef VECTORS
}
{
int vectors = 2 << 20;
int vectors_actual = vectors;
int gangs_min, gangs_max, workers_min, workers_max, vectors_min, vectors_max;
gangs_min = workers_min = vectors_min = INT_MAX;
gangs_max = workers_max = vectors_max = INT_MIN;
#pragma acc parallel copy (vectors_actual)  vector_length (vectors)
{
if (acc_on_device (acc_device_host))
{
vectors_actual = 1;
}
else if (acc_on_device (acc_device_nvidia))
{
vectors_actual = 32;
}
else
__builtin_abort ();
#pragma acc loop vector reduction (min: gangs_min, workers_min, vectors_min) reduction (max: gangs_max, workers_max, vectors_max)
for (int i = 100 * vectors_actual; i > -100 * vectors_actual; --i)
{
gangs_min = gangs_max = acc_gang ();
workers_min = workers_max = acc_worker ();
vectors_min = vectors_max = acc_vector ();
}
}
if (vectors_actual < 1)
__builtin_abort ();
if (gangs_min != 0 || gangs_max != 0
|| workers_min != 0 || workers_max != 0
|| vectors_min != 0 || vectors_max != vectors_actual - 1)
__builtin_abort ();
}
{
int gangs = 12345;
if (acc_get_device_type () == acc_device_nvidia)
gangs = 3;
int gangs_actual = gangs;
#define WORKERS 3
int workers_actual = WORKERS;
#define VECTORS 11
int vectors_actual = VECTORS;
int gangs_min, gangs_max, workers_min, workers_max, vectors_min, vectors_max;
gangs_min = workers_min = vectors_min = INT_MAX;
gangs_max = workers_max = vectors_max = INT_MIN;
#pragma acc parallel copy (gangs_actual, workers_actual, vectors_actual)  num_gangs (gangs) num_workers (WORKERS) vector_length (VECTORS)
{
if (acc_on_device (acc_device_host))
{
gangs_actual = 1;
workers_actual = 1;
vectors_actual = 1;
}
else if (acc_on_device (acc_device_nvidia))
{
vectors_actual = 32;
}
else
__builtin_abort ();
#pragma acc loop gang reduction (min: gangs_min, workers_min, vectors_min) reduction (max: gangs_max, workers_max, vectors_max)
for (int i = 100 * gangs_actual; i > -100 * gangs_actual; --i)
#pragma acc loop worker reduction (min: gangs_min, workers_min, vectors_min) reduction (max: gangs_max, workers_max, vectors_max)
for (int j = 100 * workers_actual; j > -100 * workers_actual; --j)
#pragma acc loop vector reduction (min: gangs_min, workers_min, vectors_min) reduction (max: gangs_max, workers_max, vectors_max)
for (int k = 100 * vectors_actual; k > -100 * vectors_actual; --k)
{
gangs_min = gangs_max = acc_gang ();
workers_min = workers_max = acc_worker ();
vectors_min = vectors_max = acc_vector ();
}
}
if (gangs_min != 0 || gangs_max != gangs_actual - 1
|| workers_min != 0 || workers_max != workers_actual - 1
|| vectors_min != 0 || vectors_max != vectors_actual - 1)
__builtin_abort ();
#undef VECTORS
#undef WORKERS
}
{
int gangs_min, gangs_max, workers_min, workers_max, vectors_min, vectors_max;
gangs_min = workers_min = vectors_min = INT_MAX;
gangs_max = workers_max = vectors_max = INT_MIN;
#pragma acc kernels
{
asm volatile ("" : : : "memory");
#pragma acc loop reduction (min: gangs_min, workers_min, vectors_min) reduction (max: gangs_max, workers_max, vectors_max)
for (int i = 100; i > -100; --i)
{
gangs_min = gangs_max = acc_gang ();
workers_min = workers_max = acc_worker ();
vectors_min = vectors_max = acc_vector ();
}
}
if (gangs_min != 0 || gangs_max != 1 - 1
|| workers_min != 0 || workers_max != 1 - 1
|| vectors_min != 0 || vectors_max != 1 - 1)
__builtin_abort ();
}
{
int gangs = 5;
#define WORKERS 5
#define VECTORS 13
int gangs_min, gangs_max, workers_min, workers_max, vectors_min, vectors_max;
gangs_min = workers_min = vectors_min = INT_MAX;
gangs_max = workers_max = vectors_max = INT_MIN;
#pragma acc kernels num_gangs (gangs) num_workers (WORKERS) vector_length (VECTORS)
{
asm volatile ("" : : : "memory");
#pragma acc loop reduction (min: gangs_min, workers_min, vectors_min) reduction (max: gangs_max, workers_max, vectors_max)
for (int i = 100; i > -100; --i)
{
gangs_min = gangs_max = acc_gang ();
workers_min = workers_max = acc_worker ();
vectors_min = vectors_max = acc_vector ();
}
}
if (gangs_min != 0 || gangs_max != 1 - 1
|| workers_min != 0 || workers_max != 1 - 1
|| vectors_min != 0 || vectors_max != 1 - 1)
__builtin_abort ();
#undef VECTORS
#undef WORKERS
}
return 0;
}
