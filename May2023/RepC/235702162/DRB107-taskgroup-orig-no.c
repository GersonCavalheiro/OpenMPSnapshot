#include <stdio.h>
#include <assert.h>
#include <unistd.h>
int main()
{
int result = 0;
#pragma omp parallel
{
#pragma omp single
{
#pragma omp taskgroup
{
#pragma omp task
{
sleep(3);
result = 1; 
}
}
#pragma omp task
{
result = 2; 
}
}
}
printf ("result=%d\n", result);
assert (result==2);
return 0;
}
