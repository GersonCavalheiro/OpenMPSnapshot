#include <assert.h>
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
(((void)sizeof ((x==100) ? 1 : 0)), ({
if (x==100)
{
;
}
else
{
__assert_fail("x==100", "DRB110-ordered-orig-no.c", 57, __PRETTY_FUNCTION__);
}
}));
printf("x=%d\n", x);
_ret_val_0=0;
return _ret_val_0;
}
