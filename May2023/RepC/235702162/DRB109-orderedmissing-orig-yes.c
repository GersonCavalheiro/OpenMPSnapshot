#include <stdio.h>
int main()
{
int x = 0;
int _ret_val_0;
{
int i = 0;
#pragma loop name main#0 
#pragma cetus reduction(+: x) 
for (; i<100;  ++ i)
{
x ++ ;
}
}
printf("x=%d\n", x);
_ret_val_0=0;
return _ret_val_0;
}
