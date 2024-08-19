#include <omp.h>
#include <stdio.h>
#include <assert.h>
int main(int argc, char *argv[])
{
int i;
int s = 0;
#pragma omp parallel for reduction(+:s)
for (i = 0; i < 100; i++)
{
s += i;
}
assert(s == 4950);
return 0;
}
