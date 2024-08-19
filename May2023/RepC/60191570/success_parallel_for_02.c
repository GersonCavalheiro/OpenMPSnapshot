#include <stdio.h>
#define NUM_ELEMS 1000
int main(int argc, char *argv[])
{
int c[NUM_ELEMS], x;
#pragma omp parallel for private(x)
for (int i = 0; i < NUM_ELEMS; i++)
{
x = i;
c[i] = i;
}
{
int i;
for (i = 0; i < NUM_ELEMS; i++)
{
if (c[i] != i)
abort();
}
}
return 0;
}
