#include <stdlib.h>
#include <stdio.h>
int main(int argc, char * argv[])
{
int i;
int tmp;
int len = 100;
int a[100];
int _ret_val_0;
#pragma cetus private(i) 
#pragma loop name main#0 
#pragma cetus parallel 
#pragma omp parallel for private(i)
for (i=0; i<len; i ++ )
{
a[i]=i;
}
#pragma cetus private(i, tmp) 
#pragma loop name main#1 
#pragma cetus parallel 
#pragma omp parallel for private(i, tmp)
for (i=0; i<len; i ++ )
{
tmp=(a[i]+i);
a[i]=tmp;
}
printf("a[50]=%d\n", a[50]);
_ret_val_0=0;
return _ret_val_0;
}
