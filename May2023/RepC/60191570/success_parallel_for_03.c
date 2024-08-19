#include <stdlib.h>
#include <stdio.h>
#define NUM_ELEMS 100
int main(int argc, char *argv[])
{
int i;
int s = 0;
#pragma omp parallel for reduction(+:s)
for (i = 0; i < NUM_ELEMS; i++)
{
s = s + i;
}
if (s != ((NUM_ELEMS - 1) * NUM_ELEMS)/2)
{
fprintf(stderr, "s == %d != %d\n", s, ((NUM_ELEMS - 1) * NUM_ELEMS)/2);
abort();
}
return 0;
}
