#include <stdio.h>
int main()
{
int len = 100;
double a[len], b[len], c[len];
int i, j = 0;
int _ret_val_0;
#pragma cetus private(i) 
#pragma loop name main#0 
#pragma cetus parallel 
#pragma omp parallel for private(i)
for (i=0; i<len; i ++ )
{
a[i]=(((double)i)/2.0);
b[i]=(((double)i)/3.0);
c[i]=(((double)i)/7.0);
}
#pragma cetus private(i) 
#pragma loop name main#1 
#pragma cetus reduction(+: c[i+j]) 
#pragma cetus parallel 
#pragma omp parallel for private(i) reduction(+: c[i+j])
for (i=0; i<len; i ++ )
{
c[i+j]+=(a[i]*b[i]);
}
j+=len;
printf("c[50]=%f\n", c[50]);
_ret_val_0=0;
return _ret_val_0;
}
