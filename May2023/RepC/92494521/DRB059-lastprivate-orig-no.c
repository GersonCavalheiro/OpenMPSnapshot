#include <stdio.h>
void foo()
{
int i,x;
#pragma omp parallel for private (i) lastprivate (x)
for (i=0;i<100;i++)
x=i;
printf("x=%d",x);
}
int main()
{
foo();
return 0;
}
