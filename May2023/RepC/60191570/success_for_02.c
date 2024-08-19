#include <string.h>
#define MAX_ELEMS 100
int main(int argc, char* argv[])
{
int c[MAX_ELEMS];
memset(c, 0, sizeof(c));
#pragma omp for
for (int i = 0; i < MAX_ELEMS; i++)
{
c[i] = i;
}
{
int i;
for (i = 0; i < MAX_ELEMS; i++)
{
if (c[i] != i)
abort();
}
}
return 0;
}
