#include <stdio.h>
int main (void)
{
int a=0;
#pragma omp parallel for reduction(+:a) 
for (int i = 0; i < 100; i++)
{
a += 1;
}
printf ("a=%d\n",a);
return 0;
}
