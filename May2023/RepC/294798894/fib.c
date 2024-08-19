#include <stdio.h>
#include <omp.h>
int fib(int n)
{
int i, j, r;
omp_lock_t *lock[n];
omp_init_lock(&lock[n]);
omp_set_lock( &(lock[n]) );
if (n<2)
return n;
else
{
#pragma omp task shared(i) firstprivate(n)
i=fib(n-1);
#pragma omp task shared(j) firstprivate(n)
j=fib(n-2);
#pragma omp taskwait
r = i+j;
}
omp_unset_lock( &(lock[n]) );
return r;
}
int main()
{
int n;
printf("Done by Maitreyee\n\n\n");
printf("Enter a number: ");
scanf("%d", &n);
printf("\n\n");
omp_set_dynamic(0);
omp_set_num_threads(4);
#pragma omp parallel shared(n)
{
#pragma omp single
printf ("fib(%d) = %d\n", n, fib(n));
}
}
