#include <stdlib.h>
int main(int argc, char * argv[])
{
int i;
int tmp;
int len = 100;
int a[len];
int _ret_val_0;
tmp=10;
if (argc>1)
{
len=atoi(argv[1]);
}
#pragma cetus private(i) 
#pragma loop name main#0 
for (i=0; i<len; i ++ )
{
a[i]=tmp;
tmp=(a[i]+i);
}
#pragma cetus private(i) 
#pragma loop name main#1 
for (i=0; i<len; i ++ )
{
printf("%d\n", a[i]);
}
_ret_val_0=0;
return _ret_val_0;
}
