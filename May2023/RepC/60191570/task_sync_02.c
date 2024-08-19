void f(void)
{
#pragma omp task
{
printf("1");
#pragma omp task
printf("2");
}
#pragma omp task
printf("3");
#pragma omp taskwait
}
