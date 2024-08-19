extern int i;
#pragma acc declare create(i)
void
f_omp (void)
{
#pragma omp parallel
{
#pragma acc parallel 
;
#pragma acc kernels 
;
#pragma acc data 
;
#pragma acc update host(i) 
#pragma acc enter data copyin(i) 
#pragma acc exit data delete(i) 
#pragma acc loop 
for (i = 0; i < 2; ++i)
;
}
#pragma omp for
for (i = 0; i < 3; i++)
{
#pragma acc parallel 
;
#pragma acc kernels 
;
#pragma acc data 
;
#pragma acc update host(i) 
#pragma acc enter data copyin(i) 
#pragma acc exit data delete(i) 
#pragma acc loop 
for (i = 0; i < 2; ++i)
;
}
#pragma omp sections
{
{
#pragma acc parallel 
;
}
#pragma omp section
{
#pragma acc kernels 
;
}
#pragma omp section
{
#pragma acc data 
;
}
#pragma omp section
{
#pragma acc update host(i) 
}
#pragma omp section
{
#pragma acc enter data copyin(i) 
}
#pragma omp section
{
#pragma acc exit data delete(i) 
}
#pragma omp section
{
#pragma acc loop 
for (i = 0; i < 2; ++i)
;
}
}
#pragma omp single
{
#pragma acc parallel 
;
#pragma acc kernels 
;
#pragma acc data 
;
#pragma acc update host(i) 
#pragma acc enter data copyin(i) 
#pragma acc exit data delete(i) 
#pragma acc loop 
for (i = 0; i < 2; ++i)
;
}
#pragma omp task
{
#pragma acc parallel 
;
#pragma acc kernels 
;
#pragma acc data 
;
#pragma acc update host(i) 
#pragma acc enter data copyin(i) 
#pragma acc exit data delete(i) 
#pragma acc loop 
for (i = 0; i < 2; ++i)
;
}
#pragma omp master
{
#pragma acc parallel 
;
#pragma acc kernels 
;
#pragma acc data 
;
#pragma acc update host(i) 
#pragma acc enter data copyin(i) 
#pragma acc exit data delete(i) 
#pragma acc loop 
for (i = 0; i < 2; ++i)
;
}
#pragma omp critical
{
#pragma acc parallel 
;
#pragma acc kernels 
;
#pragma acc data 
;
#pragma acc update host(i) 
#pragma acc enter data copyin(i) 
#pragma acc exit data delete(i) 
#pragma acc loop 
for (i = 0; i < 2; ++i)
;
}
#pragma omp ordered
{
#pragma acc parallel 
;
#pragma acc kernels 
;
#pragma acc data 
;
#pragma acc update host(i) 
#pragma acc enter data copyin(i) 
#pragma acc exit data delete(i) 
#pragma acc loop 
for (i = 0; i < 2; ++i)
;
}
#pragma omp target
{
#pragma acc parallel 
;
#pragma acc kernels 
;
#pragma acc data 
;
#pragma acc update host(i) 
#pragma acc enter data copyin(i) 
#pragma acc exit data delete(i) 
#pragma acc loop 
for (i = 0; i < 2; ++i)
;
}
}
void
f_acc_parallel (void)
{
#pragma acc parallel
{
#pragma omp parallel 
;
}
#pragma acc parallel
{
#pragma omp for 
for (i = 0; i < 3; i++)
;
}
#pragma acc parallel
{
#pragma omp sections 
{
;
}
}
#pragma acc parallel
{
#pragma omp single 
;
}
#pragma acc parallel
{
#pragma omp task 
;
}
#pragma acc parallel
{
#pragma omp master 
;
}
#pragma acc parallel
{
#pragma omp critical 
;
}
#pragma acc parallel
{
#pragma omp ordered 
;
}
#pragma acc parallel
{
#pragma omp target 
;
#pragma omp target data map(i) 
;
#pragma omp target update to(i) 
}
}
void
f_acc_kernels (void)
{
#pragma acc kernels
{
#pragma omp parallel 
;
}
#pragma acc kernels
{
#pragma omp for 
for (i = 0; i < 3; i++)
;
}
#pragma acc kernels
{
#pragma omp sections 
{
;
}
}
#pragma acc kernels
{
#pragma omp single 
;
}
#pragma acc kernels
{
#pragma omp task 
;
}
#pragma acc kernels
{
#pragma omp master 
;
}
#pragma acc kernels
{
#pragma omp critical 
;
}
#pragma acc kernels
{
#pragma omp ordered 
;
}
#pragma acc kernels
{
#pragma omp target 
;
#pragma omp target data map(i) 
;
#pragma omp target update to(i) 
}
}
void
f_acc_data (void)
{
#pragma acc data
{
#pragma omp parallel 
;
}
#pragma acc data
{
#pragma omp for 
for (i = 0; i < 3; i++)
;
}
#pragma acc data
{
#pragma omp sections 
{
;
}
}
#pragma acc data
{
#pragma omp single 
;
}
#pragma acc data
{
#pragma omp task 
;
}
#pragma acc data
{
#pragma omp master 
;
}
#pragma acc data
{
#pragma omp critical 
;
}
#pragma acc data
{
#pragma omp ordered 
;
}
#pragma acc data
{
#pragma omp target 
;
#pragma omp target data map(i) 
;
#pragma omp target update to(i) 
}
}
#pragma acc routine
void
f_acc_loop (void)
{
#pragma acc loop
for (i = 0; i < 2; ++i)
{
#pragma omp parallel 
;
}
#pragma acc loop
for (i = 0; i < 2; ++i)
{
#pragma omp for 
for (i = 0; i < 3; i++)
;
}
#pragma acc loop
for (i = 0; i < 2; ++i)
{
#pragma omp sections 
{
;
}
}
#pragma acc loop
for (i = 0; i < 2; ++i)
{
#pragma omp single 
;
}
#pragma acc loop
for (i = 0; i < 2; ++i)
{
#pragma omp task 
;
}
#pragma acc loop
for (i = 0; i < 2; ++i)
{
#pragma omp master 
;
}
#pragma acc loop
for (i = 0; i < 2; ++i)
{
#pragma omp critical 
;
}
#pragma acc loop
for (i = 0; i < 2; ++i)
{
#pragma omp ordered 
;
}
#pragma acc loop
for (i = 0; i < 2; ++i)
{
#pragma omp target 
;
#pragma omp target data map(i) 
;
#pragma omp target update to(i) 
}
}
#pragma acc routine
void
f_acc_routine (void)
{
#pragma omp target 
;
}
