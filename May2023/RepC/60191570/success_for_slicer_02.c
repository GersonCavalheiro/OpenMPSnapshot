#include <stdlib.h>
#include <stdio.h>
int main(int argc, char *argv[])
{
int i;
int N = 100;
int s = 0;
#pragma omp for reduction(+:s) schedule(ompss_dynamic)
for (i = 1; i <= N; i++)
{
s += i;
}
if (s != 5050)
{
fprintf(stderr, "s is %d but should be 5050\n", s);
abort();
}
return 0;
}
