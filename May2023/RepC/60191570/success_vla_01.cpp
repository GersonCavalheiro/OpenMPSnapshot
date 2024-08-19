#include <stdlib.h>
void f(int n)
{
if (n <= 0)
return;
int x[n][n];
int y[n][n];
for (int i = 0; i < n; i++)
{
for (int j = 0; j < n; j++)
{
x[i][j] = j - i;
y[i][j] = j - i;
}
}
#pragma omp task shared(x) firstprivate(y) no_copy_deps
{
for (int i = 0; i < n; i++)
{
for (int j = 0; j < n; j++)
{
x[i][j] *= 2;
y[i][j] *= 2;
}
}
}
#pragma omp taskwait
for (int i = 0; i < n; i++)
{
for (int j = 0; j < n; j++)
{
if (x[i][j] != (2*(j - i)))
{
abort();
}
if (y[i][j] != (j - i))
{
abort();
}
}
}
}
int main(int argc, char *argv[])
{
f(10);
return 0;
}
