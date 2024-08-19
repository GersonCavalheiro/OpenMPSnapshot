#include <stdio.h>
#include <assert.h>
int main()
{
int i, error;
int len = 1000;
int a[len], b = 5;
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
#pragma cetus parallel 
#pragma omp parallel for private(i)
for (i=0; i<len; i ++ )
{
a[i]=(b+(a[i]*5));
}
error=(a[9]+1);
(((void)sizeof ((error==51) ? 1 : 0)), ({
if (error==51)
{
;
}
else
{
__assert_fail("error == 51", "DRB104-nowait-barrier-orig-no.c", 69, __PRETTY_FUNCTION__);
}
}));
printf("error = %d\n", error);
_ret_val_0=0;
return _ret_val_0;
}
