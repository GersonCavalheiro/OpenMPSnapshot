#include <stdio.h>
int a=0, b=0, c=0, d=0;
int main()
{
#pragma analysis_check assert live_in(a, b) live_out(a, c)
#pragma omp sections firstprivate(b) lastprivate(c) private(d)
{
#pragma omp section
{
a = 5;
b = 5;
}
#pragma omp section
{
c = 5;
d = 5;
}
}
#pragma omp barrier
printf("a = %d\n", a);
printf("b = %d\n", b);
printf("c = %d\n", c);
printf("d = %d\n", d);
return 1;
}