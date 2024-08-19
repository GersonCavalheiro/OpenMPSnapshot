#include <omp.h>
#include <stdio.h>
void sub()
{
int j, k, a;
#pragma omp parallel num_threads(2)
{
#pragma omp for collapse(2) ordered private(j,k) schedule(static,1)
for (k=1; k<=80; k++)
for (j=1; j<=75; j++)
{
#pragma omp ordered
printf("%d %d %d\n", omp_get_thread_num(), k, j);
}
}
}
void sub2()
{
int j, k, a;
#pragma omp parallel num_threads(2)
{
#pragma omp for ordered private(j,k) schedule(static,1)
for (k=1; k<=80; k++)
for (j=1; j<=75; j++)
{
#pragma omp ordered
printf("%d %d %d\n", omp_get_thread_num(), k, j);
}
}
}
void serial()
{
int j, k, a;
for (k=1; k<=80; k++)
for (j=1; j<=75; j++)
{
printf("%d %d %d\n", omp_get_thread_num(), k, j);
}
}
int main()
{
double start_time, run_time;
printf("Ordered clause used along with the collapse clause\n");
start_time = omp_get_wtime();
sub();
run_time = omp_get_wtime() - start_time;
printf("Time to compute(in parallel) : %f\n", run_time);
printf("Ordered clause used without the collapse clause\n");
start_time = omp_get_wtime();
sub2();
run_time = omp_get_wtime() - start_time;
printf("Time to compute(in parallel) : %f\n", run_time);
return 0;
}
