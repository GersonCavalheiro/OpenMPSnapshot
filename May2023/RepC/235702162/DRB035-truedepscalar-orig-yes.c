#include <stdlib.h>
#include <stdio.h>
int main(int argc, char * argv[])
{
int i;
int tmp;
int len = 100;
int a[100];
int _ret_val_0;
tmp=10;
#pragma cetus private(i) 
#pragma loop name main#0 
for (i=0; i<len; i ++ )
{
a[i]=tmp;
tmp=(a[i]+i);
}
printf("a[50]=%d\n", a[50]);
_ret_val_0=0;
return _ret_val_0;
}
