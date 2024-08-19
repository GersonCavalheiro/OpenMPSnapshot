#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#pragma omp task
void f(int n, int m, int v[n][m]);
int g(int n1, int m1)
{
int v[n1][m1];
memset(v, 0, sizeof(v));
fprintf(stderr, "g: %p -> [%d][%d] -> %p\n", v, n1-1, m1-1, &v[n1-1][m1-1]);
f(n1, m1, v);
#pragma omp taskwait
if (v[n1-1][m1-1] != 42)
{
fprintf(stderr, "%d != 42\n", v[n1-1][m1-1]);
abort();
}
}
void f(int n1, int m1, int v[n1][m1])
{
fprintf(stderr, "f: %p -> [%d][%d] -> %p\n", v, n1-1, m1-1, &v[n1-1][m1-1]);
v[n1-1][m1-1] = 42;
}
int main(int argc, char* argv[])
{
g(10, 20);
return 0;
}
