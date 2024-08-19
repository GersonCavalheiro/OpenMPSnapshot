#include <stdio.h>
#include <stdlib.h>
int main(int argc, char * argv[])
{
int len = 100;
int a[len];
int i, x = 10;
int _ret_val_0;
if (argc>1)
{
len=atoi(argv[1]);
}
#pragma cetus private(i) 
#pragma loop name main#0 
for (i=0; i<len; i ++ )
{
a[i]=x;
x=i;
}
printf("x=%d, a[0]=%d\n", x, a[0]);
_ret_val_0=0;
return _ret_val_0;
}
