#include <stdio.h>
int a[100];
int main()
{
int len = 100;
int i, x = 10;
int _ret_val_0;
#pragma cetus private(i) 
#pragma loop name main#0 
for (i=0; i<len; i ++ )
{
a[i]=x;
x=i;
}
printf("x=%d", x);
_ret_val_0=0;
return _ret_val_0;
}
