#include <stdio.h>
#include <math.h>
#define MSIZE 200
int n=MSIZE, m=MSIZE, mits=1000;
double tol=0.0000000001, relax = 1.0, alpha = 0.0543;
double u[MSIZE][MSIZE], f[MSIZE][MSIZE], uold[MSIZE][MSIZE];
double dx, dy;
void
initialize ()
{
int i, j, xx, yy;
dx = 2.0 / (n - 1);
dy = 2.0 / (m - 1);
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
void
jacobi ()
{
double omega;
int i, j, k;
double error, resid,  ax, ay, b;
omega = relax;
dx = 2.0 / (n - 1); 
dy = 2.0 / (m - 1);
ax = 1.0 / (dx * dx);         
ay = 1.0 / (dy * dy);         
b = -2.0 / (dx * dx) - 2.0 / (dy * dy) - alpha;       
error = 10.0 * tol;
k = 1;
while (k <= mits)
{
error = 0.0;
#pragma omp parallel
{
#pragma omp for private(i,j)
for (i = 0; i < n; i++)
for (j = 0; j < m; j++)
uold[i][j] = u[i][j];
#pragma omp for private(i,j,resid) reduction(+:error) nowait
for (i = 1; i < (n - 1); i++)
for (j = 1; j < (m - 1); j++)
{
resid = (ax * (uold[i - 1][j] + uold[i + 1][j])
+ ay * (uold[i][j - 1] + uold[i][j + 1]) +
b * uold[i][j] - f[i][j]) / b;
u[i][j] = uold[i][j] - omega * resid;
error = error + resid * resid;
}
}
k = k + 1;
error = sqrt (error) / (n * m);
}                           
printf ("Total Number of Iterations:%d\n", k);
printf ("Residual:%E\n", error);
}
int main()
{
initialize();
jacobi();
return 0;
}
