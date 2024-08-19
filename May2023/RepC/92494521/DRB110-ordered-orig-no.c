#include <assert.h> 
#include <stdio.h>
int main()
{
int x =0;
#pragma omp parallel for ordered 
for (int i = 0; i < 100; ++i) {
#pragma omp ordered
{
x++;
}
}
assert (x==100);
printf ("x=%d\n",x);
return 0;
} 
