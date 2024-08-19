#include <math.h>

#include "nrutil.h"

static double at,bt,ct;
#define PYTHAG(a,b) ((at=fabs(a)) > (bt=fabs(b)) ? \
(ct=bt/at,at*sqrt(1.0+ct*ct)) : (bt ? (ct=at/bt,bt*sqrt(1.0+ct*ct)): 0.0))

static double maxarg1d,maxarg2d;
#define MAXD(a,b) (maxarg1d=(a),maxarg2d=(b),(maxarg1d) > (maxarg2d) ?\
(maxarg1d) : (maxarg2d))
#define SIGN(a,b) ((b) >= 0.0 ? fabs(a) : -fabs(a))

void svdcmp_d(double **a,int m,int n, double *w, double **v)
{
int flag,i,its,j,jj,k,l,nm;
double c,f,h,s,x,y,z;
double anorm=0.0,g=0.0,scale=0.0;
double *rv1;

if (m < n) nrerror("SVDCMP: You must augment A with extra zero rows");
rv1=dvector(1,n);

#pragma omp parallel for num_threads(NUM_THREADS) if(n > 100)
for (i=1;i<=n;i++) {
l=i+1;
rv1[i]=scale*g;
g=s=scale=0.0;
if (i <= m) {
#pragma omp parallel for num_threads(NUM_THREADS) if(m > 100)
for (k=i;k<=m;k++) scale += fabs(a[k][i]);
if (scale) {
for (k=i;k<=m;k++) {
a[k][i] /= scale;
s += a[k][i]*a[k][i];
}
g = -SIGN(sqrt(s),a[i][i]);
h=a[i][i]*g-s;
a[i][i]=a[i][i]-g;
if (i != n) {
for (j=l;j<=n;j++) {
for (s=0.0,k=i;k<=m;k++) s += a[k][i]*a[k][j];
for (k=i;k<=m;k++) a[k][j] += (s/h)*a[k][i];
}
}
for (k=i;k<=m;k++) a[k][i] /= (1.0/scale);
}
}
w[i]=scale*g;
g=s=scale=0.0;
if (i <= m && i != n) {
for (k=l;k<=n;k++) scale += fabs(a[i][k]);
if (scale) {
for (k=l;k<=n;k++) {
s += a[i][k]*a[i][k]/(scale*scale);
a[i][k] /= scale;
}
f=a[i][l];
g = -SIGN(sqrt(s),f);
a[i][l]=f-g;
for (k=l;k<=n;k++) rv1[k]=a[i][k]/(f*g-s);
if (i != m) {
for (j=l;j<=m;j++) {
for (s=0.0,k=l;k<=n;k++) s += a[j][k]*a[i][k];
for (k=l;k<=n;k++) a[j][k] += s*rv1[k];
}
}
for (k=l;k<=n;k++) a[i][k] *= scale;
}
}
anorm=MAXD(anorm,(fabs(w[i])+fabs(rv1[i])));
}
for (i=n;i>=1;i--) {
if (i < n) {
if (g) {
for (j=l;j<=n;j++)
v[j][i]=(a[i][j]/a[i][l])/g;
for (j=l;j<=n;j++) {
for (s=0.0,k=l;k<=n;k++) s += a[i][k]*v[k][j];
for (k=l;k<=n;k++) v[k][j] += s*v[k][i];
}
}
for (j=l;j<=n;j++) v[i][j]=v[j][i]=0.0;
}
v[i][i]=1.0;
g=rv1[i];
l=i;
}
for (i=n;i>=1;i--) {
l=i+1;
g=w[i];
if (i < n)
for (j=l;j<=n;j++) a[i][j]=0.0;
if (g) {
g=1.0/g;
if (i != n) {
for (j=l;j<=n;j++) {
for (s=0.0,k=l;k<=m;k++) s += a[k][i]*a[k][j];
f=(s/a[i][i])*g;
for (k=i;k<=m;k++) a[k][j] += f*a[k][i];
}
}
for (j=i;j<=m;j++) a[j][i] *= g;
} else {
for (j=i;j<=m;j++) a[j][i]=0.0;
}
++a[i][i];
}
for (k=n;k>=1;k--) {
for (its=1;its<=30;its++) {
flag=1;
for (l=k;l>=1;l--) {
nm=l-1;
if (fabs(rv1[l])+anorm == anorm) {
flag=0;
break;
}
if (fabs(w[nm])+anorm == anorm) break;
}
if (flag) {
s=1.0;
for (i=l;i<=k;i++) {
if (fabs(f)+anorm != anorm) {
g=w[i];
w[i]=PYTHAG(s*rv1[i],w[i]);
for (j=1;j<=m;j++) {
y=a[j][nm];
z=a[j][i];
a[j][nm]=y*(g/w[i])+z*(-s*rv1[i]/w[i]);
a[j][i]=z*(g/w[i])-y*(-s*rv1[i]/w[i]);
}
}
}
}
z=w[k];
if (l == k) {
if (z < 0.0) {
w[k] = -z;
for (j=1;j<=n;j++) v[j][k]=(-v[j][k]);
}
break;
}
if (its == 30) nrerror("No convergence in 30 SVDCMP iterations");
x=w[l];
nm=k-1;
y=w[nm];
g=rv1[nm];
h=rv1[k];
f=((y-z)*(y+z)+(g-h)*(g+h))/(2.0*h*y);
g=PYTHAG(f,1.0);
f=((x-z)*(x+z)+h*((y/(f+SIGN(g,f)))-h))/x;
c=s=1.0;
for (j=l;j<=nm;j++) {
i=j+1;
g=rv1[i];
y=w[i];
h=s*g;
g=c*g;
z=PYTHAG(f,h);
rv1[j]=z;
c=f/z;
s=h/z;
f=x*c+g*s;
g=g*c-x*s;
h=y*s;
y=y*c;
for (jj=1;jj<=n;jj++) {
x=v[jj][j];
z=v[jj][i];
v[jj][j]=x*c+z*s;
v[jj][i]=z*c-x*s;
}
z=PYTHAG(f,h);
w[j]=z;
if (z) {
z=1.0/z;
c=f*z;
s=h*z;
}
f=(c*g)+(s*y);
x=(c*y)-(s*g);
for (jj=1;jj<=m;jj++) {
y=a[jj][j];
z=a[jj][i];
a[jj][j]=y*c+z*s;
a[jj][i]=z*c-y*s;
}
}
rv1[l]=0.0;
rv1[k]=f;
w[k]=x;
}
}
free_dvector(rv1,1,n);
}

#undef SIGN
#undef MAXD
#undef PYTHAG
