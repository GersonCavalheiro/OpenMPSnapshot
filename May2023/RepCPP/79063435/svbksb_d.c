#define NRANSI

#include "nrutil.h"
void svbksb_d(double **u, double w[], double **v, int m, int n, double b[], double x[])
{
int jj,j,i;
double s,*tmp;

tmp=dvector(1,n);
# pragma omp parallel for if(n > 100) \
shared(n, m, u, b, w, tmp) \
private(j, s, i)
for (j=1;j<=n;j++) {
s=0.0;
if (w[j]) {
# pragma omp parallel for if(m > 100)
for (i=1;i<=m;i++) s += u[i][j]*b[i];
s /= w[j];
}
tmp[j]=s;
}
# pragma omp parallel for if(n > 100)\
shared(n, v, tmp, x) \
private(j, jj, s)
for (j=1;j<=n;j++) {
s=0.0;
for (jj=1;jj<=n;jj++) s += v[j][jj]*tmp[jj];
x[j]=s;
}
free_dvector(tmp,1,n);
}
#undef NRANSI
