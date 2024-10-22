#include <stdio.h>
#include <math.h>
#define MSIZE 200
int n=MSIZE, m=MSIZE;
double alpha = 0.0543;
double u[MSIZE][MSIZE], f[MSIZE][MSIZE], uold[MSIZE][MSIZE];
double dx, dy;
void
initialize ()
{
int i, j, xx, yy;
dx = 2.0 / (n - 1);
dy = 2.0 / (m - 1);
#pragma omp parallel for private(i,j,xx,yy)
for (i = 0; i < n; i++)
for (j = 0; j < m; j++)
{
xx = (int) (-1.0 + dx * (i - 1));       
yy = (int) (-1.0 + dy * (j - 1));       
u[i][j] = 0.0;
f[i][j] = -1.0 * alpha * (1.0 - xx * xx) * (1.0 - yy * yy)
- 2.0 * (1.0 - xx * xx) - 2.0 * (1.0 - yy * yy);
}
}
int main()
{
initialize();
return 0;
}
