#include <stdio.h>
void foo()
{
int i, x;
#pragma cetus private(i) 
#pragma cetus lastprivate(x) 
#pragma loop name foo#0 
#pragma cetus parallel 
#pragma omp parallel for private(i) lastprivate(x)
for (i=0; i<100; i ++ )
{
x=i;
}
printf("x=%d", x);
return ;
}
int main()
{
int _ret_val_0;
foo();
_ret_val_0=0;
return _ret_val_0;
}
