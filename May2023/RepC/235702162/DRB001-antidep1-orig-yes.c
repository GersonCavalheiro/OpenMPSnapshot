#include <stdio.h>
int main(int argc, char * argv[])
{
int i;
int len = 1000;
int a[1000];
int _ret_val_0;
#pragma cetus private(i) 
#pragma loop name main#0 
#pragma cetus parallel 
#pragma omp parallel for private(i)
for (i=0; i<len; i ++ )
{
a[i]=i;
}
#pragma cetus private(i) 
#pragma loop name main#1 
for (i=0; i<(len-1); i ++ )
{
a[i]=(a[i+1]+1);
}
printf("a[500]=%d\n", a[500]);
_ret_val_0=0;
return _ret_val_0;
}
