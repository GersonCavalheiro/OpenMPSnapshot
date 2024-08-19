#include<stdio.h>
int main(int argc, char * argv[])
{
int i;
int len = 100;
int a[len], b[len];
int _ret_val_0;
#pragma cetus private(i) 
#pragma loop name main#0 
#pragma cetus parallel 
#pragma omp parallel for private(i)
for (i=0; i<len; i ++ )
{
a[i]=i;
b[i]=i;
}
{
static int tmp;
#pragma cetus private(i, tmp) 
#pragma loop name main#1 
#pragma cetus parallel 
#pragma omp parallel for private(i, tmp)
for (i=0; i<len; i ++ )
{
tmp=(a[i]+i);
a[i]=tmp;
}
}
{
int tmp;
#pragma cetus private(i, tmp) 
#pragma loop name main#2 
#pragma cetus parallel 
#pragma omp parallel for private(i, tmp)
for (i=0; i<len; i ++ )
{
tmp=(b[i]+i);
b[i]=tmp;
}
}
printf("a[50]=%d b[50]=%d\n", a[50], b[50]);
_ret_val_0=0;
return _ret_val_0;
}
