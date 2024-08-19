#include <stdio.h>
#include <omp.h>
int main(int argc, char **argv)
{
const long N = 1000000;
int i, a[N];
#pragma omp parallel for
for (i = 0; i < N; i++)
a[i] = 2 * a[i];
return 0;
}
