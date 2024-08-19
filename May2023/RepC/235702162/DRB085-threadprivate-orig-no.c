#include <stdio.h>
#include <assert.h>
#include <omp.h> 
int sum0 = 0;
int sum1 = 0;
void foo(int i)
{
sum0 = sum0 + i;
}
int main()
{
int len = 1000;
int i;
int sum = 0;
{
for (i = 0; i <= len - 1; i += 1) {
foo(i);
}
{
sum = sum + sum0;
}
}
#pragma omp parallel for private (i) reduction (+:sum1) firstprivate (len)
for (i = 0; i <= len - 1; i += 1) {
sum1 = sum1 + i;
}
printf("sum=%d; sum1=%d\n",sum,sum1);
(((void )(sizeof(((sum == sum1?1 : 0))))) , ((
{
if (sum == sum1) 
;
else 
__assert_fail("sum==sum1","DRB085-threadprivate-orig-no.c",80,__PRETTY_FUNCTION__);
})));
return 0;
}
