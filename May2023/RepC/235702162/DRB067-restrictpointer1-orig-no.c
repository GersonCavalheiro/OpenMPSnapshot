#include <stdlib.h>
typedef double real8;
void foo(real8 * restrict newSxx, real8 * restrict newSyy, int length)
{
int i;
#pragma cetus private(i) 
#pragma loop name foo#0 
#pragma cetus parallel 
#pragma omp parallel for private(i)
for (i=0; i<=(length-1); i+=1)
{
newSxx[i]=0.0;
newSyy[i]=0.0;
}
return ;
}
void print(real8 * restrict newSxx, real8 * restrict newSyy, int length)
{
int i;
#pragma cetus private(i) 
#pragma loop name print#0 
for (i=0; i<=(length-1); i+=1)
{
printf("%lf %lf\n", newSxx[i], newSyy[i]);
}
return ;
}
int main()
{
int length = 1000;
real8 * newSxx = malloc(length*sizeof (real8));
real8 * newSyy = malloc(length*sizeof (real8));
int _ret_val_0;
foo(newSxx, newSyy, length);
print(newSxx, newSyy, length);
free(newSxx);
free(newSyy);
_ret_val_0=0;
return _ret_val_0;
}
