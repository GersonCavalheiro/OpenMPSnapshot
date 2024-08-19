#include <stdio.h>
double a[100][100], b[100][100], c[100][100];
int init()
{
int i, j, k;
int _ret_val_0;
#pragma cetus private(i, j) 
#pragma loop name init#0 
#pragma cetus parallel 
#pragma omp parallel for private(i, j)
for (i=0; i<100; i ++ )
{
#pragma cetus private(j) 
#pragma loop name init#0#0 
#pragma cetus parallel 
#pragma omp parallel for private(j)
for (j=0; j<100; j ++ )
{
a[i][j]=(((double)i)*j);
b[i][j]=(((double)i)*j);
c[i][j]=(((double)i)*j);
}
}
_ret_val_0=0;
return _ret_val_0;
}
int mmm()
{
int i, j, k;
int _ret_val_0;
#pragma cetus private(i, j, k) 
#pragma loop name mmm#0 
#pragma cetus parallel 
#pragma omp parallel for private(i, j, k)
for (i=0; i<100; i ++ )
{
#pragma cetus private(j, k) 
#pragma loop name mmm#0#0 
#pragma cetus reduction(+: c[i][j]) 
#pragma cetus parallel 
#pragma omp parallel for private(j, k) reduction(+: c[i][j])
for (k=0; k<100; k ++ )
{
#pragma cetus private(j) 
#pragma loop name mmm#0#0#0 
#pragma cetus parallel 
#pragma omp parallel for private(j)
for (j=0; j<100; j ++ )
{
c[i][j]=(c[i][j]+(a[i][k]*b[k][j]));
}
}
}
_ret_val_0=0;
return _ret_val_0;
}
int print()
{
int i, j, k;
int _ret_val_0;
#pragma cetus private(i, j) 
#pragma loop name print#0 
for (i=0; i<100; i ++ )
{
#pragma cetus private(j) 
#pragma loop name print#0#0 
for (j=0; j<100; j ++ )
{
printf("%lf %lf %lf\n", c[i][j], a[i][j], b[i][j]);
}
}
_ret_val_0=0;
return _ret_val_0;
}
int main()
{
int _ret_val_0;
init();
mmm();
print();
_ret_val_0=0;
return _ret_val_0;
}
