#include <stdlib.h>
#include <stdio.h>
#include <time.h>
int main(int argc, char * argv[])
{
int i;
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
srand(time((void * )0));
#pragma cetus private(i) 
#pragma loop name main#1 
for (i=0; i<(len-1); i ++ )
{
a[i+1]=(a[i]+1);
}
printf("a[50]=%d\n", a[50]);
_ret_val_0=0;
return _ret_val_0;
}
