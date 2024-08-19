#include <stdlib.h>
typedef double real8;
void foo(real8 * restrict newSxx, real8 * restrict newSyy, int length)
{
int i;
#pragma omp parallel for private (i) firstprivate (length)
for (i = 0; i <= length - 1; i += 1) {
newSxx[i] = 0.0;
newSyy[i] = 0.0;
}
}
int main()
{
int length=1000;
real8* newSxx = malloc (length* sizeof (real8));
real8* newSyy = malloc (length* sizeof (real8));
foo(newSxx, newSyy, length);
free (newSxx);
free (newSyy);
return 0;
}
