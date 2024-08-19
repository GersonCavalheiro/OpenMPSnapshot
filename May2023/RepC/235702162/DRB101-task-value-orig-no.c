#include <stdio.h>
#define MYLEN 100
int a[MYLEN];
int b[MYLEN];
void gen_task(int i)
{
a[i]= i+1;
}
int main()
{
int i=0;
#pragma omp parallel for private(i ) 
for (i=0; i<MYLEN; i++)
{
gen_task(i);
}
#pragma omp parallel for private(i ) 
for (i=0; i<MYLEN; i++)
{
if (a[i]!= i+1)
{
b[i] = a[i];
}
}
for (i=0; i<MYLEN; i++)
{
printf("%d %d\n", a[i], b[i]);
}
return 0;
}
