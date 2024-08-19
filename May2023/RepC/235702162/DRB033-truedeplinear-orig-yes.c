#include <stdlib.h>
#include <stdio.h>
int main(int argc, char * argv[])
{
int i;
int a[2000];
int _ret_val_0;
#pragma cetus private(i) 
#pragma loop name main#0 
#pragma cetus parallel 
#pragma omp parallel for private(i)
for (i=0; i<2000; i ++ )
{
a[i]=i;
}
#pragma cetus private(i) 
#pragma loop name main#1 
for (i=0; i<1000; i ++ )
{
a[(2*i)+1]=(a[i]+1);
}
printf("a[1001]=%d\n", a[1001]);
_ret_val_0=0;
return _ret_val_0;
}
