#include <stdio.h>
#include <math.h>
#include <omp.h>

void daxpy(int i, double *X, double *Y, double a)
{
if (i < (1 << 16)) 
X[i] = a * X[i] + Y[i];
}

double timeUniProcDAXPY(double *X, double *Y, double a)
{
double start = omp_get_wtime();

for (int i = 0; i < (1 << 16); ++i)
daxpy(i, X, Y, a);

return (omp_get_wtime() - start);
}

int main()
{
double a = 341; 
double X[(1 << 16)], Y[(1 << 16)];

double t1 = timeUniProcDAXPY(X, Y, a);

double max_speed_up = -1e9;
int opt_threads;
printf("Time taken for uni-processor implementation: %lf seconds\n", t1);

printf("Maximum number of threads: %d\n", omp_get_max_threads());
for (int i = 2; i <= omp_get_max_threads(); ++i)
{
omp_set_num_threads(i);
int num_threads = omp_get_num_threads();
printf("i = %d \tnum_threads: %d\n", i, num_threads);
int blk_size = (ceil)((1 << 16) / num_threads);

double start = omp_get_wtime();

#pragma omp parallel
{
int id = omp_get_thread_num();

for (int j = 0; j < blk_size; ++j)
{
long int index = j * num_threads + id;
daxpy((int)index, X, Y, a);
}
}

double time_elapsed = omp_get_wtime() - start;
double speed_up = time_elapsed / t1;

printf("Num of threads: %d\t Time elapsed: %lf \t Speed-Up: %lf\n", num_threads, omp_get_wtime() - start, speed_up);

if (speed_up > max_speed_up)
{
max_speed_up = speed_up;
opt_threads = omp_get_num_threads();
}
}

return 0;
}