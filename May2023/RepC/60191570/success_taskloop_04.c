#include <assert.h>
void array_add(int *c, int *a, int *b, int N)
{
#pragma oss taskloop grainsize(1) in(a[i - 1],b[i]) inout(c[i])
for (int i = 1; i < N; i++)
{
c[i - 1] += a[i - 1] + b[i];
}
#pragma oss taskwait
}
int a[] = { 1, 2, 3, 4 };
int b[] = { 5, 6, 7, 8 };
int c[] = { -1, -1, -1, -1 };
const int ref[] = { 6, 8, 10, -1 };
const int SIZE = sizeof(a) / sizeof(*a);
int main(int argc, char *argv[])
{
array_add(c, a, b, SIZE);
int i;
for ( i = 0; i < SIZE; i++)
{
assert(c[i] == ref[i] && "Invalid value");
}
return 0;
}
