#include <stdlib.h>
#include <string.h>
void foo(int *a, int N)
{
int i;
memset(a, 0, sizeof(a[0])* N);
#pragma hlt unroll(33)
for (i=0; i<N; i+=7)
{
a[i] = 1;
}
for (i=0; i<N; i+=7)
{
if (a[i] != 1) abort();
a[i] = 0;
}
for (i=0; i<N; i++)
{
if (a[i] != 0) abort();
}
memset(a, 0, sizeof(a[0])* N);
#pragma hlt unroll(33)
for (i=N - 1; i>=0; i+=-7)
{
a[i] = 1;
}
for (i=N - 1; i>=0; i+=-7)
{
if (a[i] != 1) abort();
a[i] = 0;
}
for (i=0; i<N; i++)
{
if (a[i] != 0) abort();
}
}
enum { SIZE = 200 };
int x[SIZE];
int main(int argc, char *argv[])
{
foo(x, SIZE);
return 0;
}
