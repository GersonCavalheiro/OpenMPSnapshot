#include <stdio.h>
#include <assert.h>
int sum0 = 0, sum1 = 0;
void foo(int i)
{
sum0=(sum0+i);
return ;
}
int main()
{
int i, sum = 0;
int _ret_val_0;
#pragma cetus private(i) 
#pragma loop name main#0 
for (i=1; i<=1000; i ++ )
{
foo(i);
}
sum=(sum+sum0);
#pragma cetus private(i) 
#pragma loop name main#1 
#pragma cetus parallel 
#pragma omp parallel for private(i)
for (i=1; i<=1000; i ++ )
{
}
sum1+=500500;
printf("sum=%d; sum1=%d\n", sum, sum1);
_ret_val_0=0;
return _ret_val_0;
}
