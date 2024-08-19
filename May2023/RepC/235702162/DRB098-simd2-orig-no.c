#include <stdio.h>
int main()
{
int len = 100;
double a[len][len], b[len][len], c[len][len];
int i, j;
int _ret_val_0;
#pragma cetus private(i, j) 
#pragma loop name main#0 
#pragma cetus parallel 
#pragma omp parallel for private(i, j)
for (i=0; i<len; i ++ )
{
#pragma cetus private(j) 
#pragma loop name main#0#0 
#pragma cetus parallel 
#pragma omp parallel for private(j)
for (j=0; j<len; j ++ )
{
a[i][j]=(((double)i)/2.0);
b[i][j]=(((double)i)/3.0);
c[i][j]=(((double)i)/7.0);
}
}
#pragma cetus private(i, j) 
#pragma loop name main#1 
#pragma cetus parallel 
#pragma omp parallel for private(i, j)
for (i=0; i<len; i ++ )
{
#pragma cetus private(j) 
#pragma loop name main#1#0 
#pragma cetus parallel 
#pragma omp parallel for private(j)
for (j=0; j<len; j ++ )
{
c[i][j]=(a[i][j]*b[i][j]);
}
}
printf("c[50][50]=%f\n", c[50][50]);
_ret_val_0=0;
return _ret_val_0;
}
