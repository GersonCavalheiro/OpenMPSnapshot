#include <stdlib.h>
#define NUM_ELEMS 1000
int main(int argc, char *argv[])
{
int i = -1;
int c[NUM_ELEMS];
#pragma omp parallel for
for (i = 0; i < NUM_ELEMS; i++)
{
c[i] = i;
}
if (i != -1)
abort();
for (i = 0; i < NUM_ELEMS; i++)
{
if (c[i] != i)
abort();
}
return 0;
}
