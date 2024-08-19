#include <stdio.h>
int main(int argc, char * argv[])
{
int i, i2;
int len = 2560;
double sum = 0.0, sum2 = 0.0;
double a[len], b[len];
int _ret_val_0;
#pragma cetus private(i) 
#pragma loop name main#0 
#pragma cetus parallel 
#pragma omp parallel for private(i)
for (i=0; i<len; i ++ )
{
a[i]=(((double)i)/2.0);
b[i]=(((double)i)/3.0);
}
#pragma cetus private(i, i2) 
#pragma loop name main#1 
#pragma cetus reduction(+: sum) 
#pragma cetus parallel 
#pragma omp parallel for private(i, i2) reduction(+: sum)
for (i2=0; i2<len; i2+=256)
{
#pragma cetus private(i) 
#pragma loop name main#1#0 
#pragma cetus reduction(+: sum) 
#pragma cetus parallel 
#pragma omp parallel for private(i) reduction(+: sum)
for (i=i2; i<(((i2+256)<len) ? (i2+256) : len); i ++ )
{
sum+=(a[i]*b[i]);
}
}
#pragma cetus private(i) 
#pragma loop name main#2 
#pragma cetus reduction(+: sum2) 
#pragma cetus parallel 
#pragma omp parallel for private(i) reduction(+: sum2)
for (i=0; i<len; i ++ )
{
sum2+=(a[i]*b[i]);
}
printf("sum=%lf sum2=%lf\n", sum, sum2);
_ret_val_0=0;
return _ret_val_0;
}
