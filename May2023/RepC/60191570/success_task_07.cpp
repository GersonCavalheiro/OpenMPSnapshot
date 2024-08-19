#include <stdlib.h>
void foo(int* a, int n)
{
#pragma omp task inout(a[0;n])
{
int i;
for (i = 0; i < n; i++)
{
a[i]++;
}
}
#pragma omp taskwait
}
int main(int argc, char *argv[])
{
const int t[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
int x[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
int *ptr = x;
foo(ptr, 10);
int i;
for (i = 0; i < 10; i++)
{
if (x[i] != t[i])
abort();
}
}
