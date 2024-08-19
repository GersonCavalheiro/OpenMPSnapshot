#include <stdio.h>
#include <omp.h>
float work1(int i)
{
return 1.0 * i;
}
float work2(int i)
{
return 2.0 * i;
}
void atomic_example(float *x, float *y, int *index, int n)
{
int i;
#pragma omp parallel for shared(x, y, index, n)
for (i=0; i<n; i++) 
{
#pragma omp atomic update
x[index[i]] += work1(i);
y[i] += work2(i);
}
printf("x[99] is %f and y[99] is %f\n",x[99],y[99]);
}
int main()
{
float x[100];
float y[100];
int index[100];
int i;
for (i = 0; i < 100; i++) 
{
index[i] = i % 5;
y[i]=index[i]/3;
}
for (i = 0; i < 100; i++)
x[i] = i/3;
atomic_example(x, y, index, 100);
return 0;
}
