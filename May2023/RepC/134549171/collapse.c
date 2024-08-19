#include <stdio.h>
#include <omp.h>
void test()
{
int j, k, jlast, klast;
#pragma omp parallel
{
#pragma omp for collapse(2) lastprivate(jlast, klast)
for (k=1; k<=2; k++)
for (j=1; j<=3; j++)
{
jlast=j;
klast=k;
}
#pragma omp single
printf("%d %d\n", klast, jlast);
}
}
int main()
{
test();
return 0;
}
