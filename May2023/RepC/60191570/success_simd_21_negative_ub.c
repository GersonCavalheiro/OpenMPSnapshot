#include <stdio.h>
int __attribute__((noinline)) test (int sizey,
int *u)
{
int sum =0;
int j;
#pragma omp simd reduction(+:sum) suitable(sizey)
for (j=1; j <= sizey-2; j++)
{
sum += u[j];
}
return sum;
}
int main( int argc, char *argv[] )
{
int i, result;
int a[101];
for(i=0; i<101; i++)
{
a[i] = 1;
}
result = test(96, a);
if (result != 94)
{
fprintf(stderr, "Error: result %d != 96\n", result);
return 1;
}
result = test(64, a);
if (result != 62)
{
fprintf(stderr, "Error: result %d != 62\n", result);
return 1;
}
result = test(16, a);
if (result != 14)
{
fprintf(stderr, "Error: result %d != 14\n", result);
return 1;
}
result = test(32, a);
if (result != 30)
{
fprintf(stderr, "Error: result %d != 30\n", result);
return 1;
}
return 0;
}
