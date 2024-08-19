#include <stdlib.h>
#include <stdio.h>
int main(int argc, char *argv[])
{
int m[3][4][5];
int i, j, k;
#pragma omp for collapse(3) shared(m)
for (i = 0; i < 3; i++)
{
for (j = 0; j < 4; j++)
{
for (k = 0; k < 5; k++)
{
m[i][j][k] = i + j + k;
}
}
}
for (i = 0; i < 3; i++)
{
for (j = 0; j < 4; j++)
{
for (k = 0; k < 5; k++)
{
if (m[i][j][k] != i + j + k)
{
fprintf(stderr, "Invalid m[%d][%d][%d] = %d != %d\n", 
i, j, k, m[i][j][k], i + j + k);
abort();
}
}
}
}
return 0;
}
