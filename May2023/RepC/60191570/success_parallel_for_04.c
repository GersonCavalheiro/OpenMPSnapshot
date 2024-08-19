#ifndef __ICC
#include <stdlib.h>
void f(int N1, int N2, int a[N1][N2])
{
int i;
#pragma omp parallel for
for (i = 0; i < N1; i++)
{
int j;
for (j = 0; j < N2; j++)
{
a[i][j] = i - j;
}
}
}
int main(int argc, char* argv[])
{
int a[10][20];
f(10, 20, a);
int i;
for (i = 0; i < 10; i++)
{
int j;
for (j = 0; j < 20; j++)
{
if (a[i][j] != (i-j))
{
abort();
}
}
}
return 0;
}
#else
int main(int argc, char* argv[])
{
return 0;
}
#endif
