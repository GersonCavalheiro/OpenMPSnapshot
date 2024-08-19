#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define MAX_ELEMS 100
int main(int argc, char *argv[])
{
int c[MAX_ELEMS];
int i;
memset(c, 0, sizeof(c));
#pragma omp for
for (i = 0; i < MAX_ELEMS; i++)
{
c[i] = i;
}
#pragma omp for
for (i = 0; i < MAX_ELEMS; i++)
{
c[i] = i;
}
for (i = 0; i < MAX_ELEMS; i++)
{
if (c[i] != i)
{
fprintf(stderr, "c[%d] == %d != %d\n", i, c[i], i);
abort();
}
}
return 0;
}
