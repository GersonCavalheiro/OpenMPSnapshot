#include <stdio.h>
#define MYLEN 100
int a[MYLEN];
void gen_task(int i)
{
#pragma omp task
{
a[i]= i+1;
}
}
int main()
{
int i=0;
#pragma omp parallel
{
#pragma omp single
{
for (i=0; i<MYLEN; i++)
{
gen_task(i);
}
}
}
for (i=0; i<MYLEN; i++)
{
if (a[i]!= i+1)
{
printf("warning: a[%d] = %d, not expected %d\n", i, a[i], i+1);
}
}
return 0;
}
