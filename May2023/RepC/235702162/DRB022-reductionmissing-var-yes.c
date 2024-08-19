#include <stdio.h>
#include <stdlib.h>
int main(int argc, char * argv[])
{
int i, j;
float temp, sum = 0.0;
int len = 100;
float u[len][len];
int _ret_val_0;
if (argc>1)
{
len=atoi(argv[1]);
}
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
u[i][j]=0.5;
}
}
#pragma cetus private(i, j, temp) 
#pragma loop name main#1 
#pragma cetus reduction(+: sum) 
#pragma cetus parallel 
#pragma omp parallel for private(i, j, temp) reduction(+: sum)
for (i=0; i<len; i ++ )
{
#pragma cetus private(j, temp) 
#pragma loop name main#1#0 
#pragma cetus reduction(+: sum) 
#pragma cetus parallel 
#pragma omp parallel for private(j, temp) reduction(+: sum)
for (j=0; j<len; j ++ )
{
temp=u[i][j];
sum=(sum+(temp*temp));
}
}
printf("sum = %f\n", sum);
_ret_val_0=0;
return _ret_val_0;
}
