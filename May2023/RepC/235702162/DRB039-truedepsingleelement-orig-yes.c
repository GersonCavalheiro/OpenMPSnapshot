#include <stdlib.h>
#include <stdio.h>
int main(int argc, char * argv[])
{
int len = 1000;
int i;
int a[1000];
int _ret_val_0;
a[0]=2;
#pragma cetus private(i) 
#pragma loop name main#0 
for (i=0; i<len; i ++ )
{
a[i]=(a[i]+a[0]);
}
printf("a[500]=%d\n", a[500]);
_ret_val_0=0;
return _ret_val_0;
}
