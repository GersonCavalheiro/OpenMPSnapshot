

#include <cstdio>
#include <omp.h>

int ser_fib(int n)
{
int x, y;
if (n < 2)
return n;

x = ser_fib(n - 1);

y = ser_fib(n - 2);

return x+y;
}

int fib(int n)
{
int x, y;
if (n < 2)
return n;
else if (n < 30)
return ser_fib(n);

#pragma omp task shared(x)
{
x = fib(n - 1);
}

#pragma omp task shared(y)
{
y = fib(n - 2);
}

#pragma omp taskwait

return x+y;

}


int main()
{
int n,fibonacci;
double starttime;
printf("Please insert n, to calculate fib(n): \n");
scanf("%d",&n);
starttime=omp_get_wtime();

#pragma omp parallel
#pragma omp single
{
fibonacci=fib(n);
}

printf("fib(%d)=%d \n",n,fibonacci);
printf("calculation took %lf sec\n",omp_get_wtime()-starttime);
return 0;
}
