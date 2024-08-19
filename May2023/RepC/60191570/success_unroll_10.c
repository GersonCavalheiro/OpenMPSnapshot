#include <stdlib.h>
#include <string.h>
void foo(int *a, int N)
{
int i;
memset(a, 0, sizeof(*a) * N);
#pragma hlt unroll(8)
for (i=0; i<N; i++)
{
int c = i;
a[i] = c;
}
for (i=0; i<N; i++)
{
if (a[i] != i) abort();
}
}
enum { SIZE = 200 };
int x[SIZE];
int main(int argc, char *argv[])
{
foo(x, SIZE);
return 0;
}
