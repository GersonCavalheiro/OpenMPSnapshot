#include <stdio.h>
#include <assert.h>
int sum0 = 0, sum1 = 0;
int main()
{
int len = 1000;
int i, sum = 0;
int _ret_val_0;
#pragma cetus private(i) 
#pragma loop name main#0 
#pragma cetus parallel 
#pragma omp parallel for private(i)
for (i=0; i<len; i ++ )
{
}
sum0+=499500;
sum=(sum+sum0);
#pragma cetus private(i) 
#pragma loop name main#1 
#pragma cetus parallel 
#pragma omp parallel for private(i)
for (i=0; i<len; i ++ )
{
}
sum1+=(((-1*len)+(len*len))/2);
printf("sum=%d; sum1=%d\n", sum, sum1);
(((void)sizeof ((sum==sum1) ? 1 : 0)), ({
if (sum==sum1)
{
;
}
else
{
__assert_fail("sum==sum1", "DRB091-threadprivate2-orig-no.c", 74, __PRETTY_FUNCTION__);
}
}));
_ret_val_0=0;
return _ret_val_0;
}
