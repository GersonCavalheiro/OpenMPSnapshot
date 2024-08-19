#include <stdio.h>
#include <math.h>
#define MSIZE 200
#include <omp.h> 
int n = 200;
int m = 200;
double alpha = 0.0543;
double u[200][200];
double f[200][200];
double uold[200][200];
double dx;
double dy;
void initialize()
{
int i;
int j;
int xx;
int yy;
dx = 2.0 / (n - 1);
dy = 2.0 / (m - 1);
#pragma omp parallel for private (xx,yy,i,j) firstprivate (n,m)
for (i = 0; i <= n - 1; i += 1) {
#pragma omp parallel for private (xx,yy,j) firstprivate (alpha,dx,dy)
for (j = 0; j <= m - 1; j += 1) {
xx = ((int )(- 1.0 + dx * (i - 1)));
yy = ((int )(- 1.0 + dy * (j - 1)));
u[i][j] = 0.0;
f[i][j] = - 1.0 * alpha * (1.0 - (xx * xx)) * (1.0 - (yy * yy)) - 2.0 * (1.0 - (xx * xx)) - 2.0 * (1.0 - (yy * yy));
}
}
}
int main()
{
initialize();
int i;
int j;
for (i = 0; i <= n - 1; i += 1) {
for (j = 0; j <= m - 1; j += 1) {
printf("%lf %lf\n",u[i][j],f[i][j]);
}
}
return 0;
}
