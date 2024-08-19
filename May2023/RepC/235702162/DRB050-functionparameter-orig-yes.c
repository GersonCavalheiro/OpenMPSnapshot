#include <stdio.h>
#include <stdlib.h>
void foo1(double o1[], double c[], int len)
{ 
int i ;
for (i = 0; i < len; ++i) {
double volnew_o8 = 0.5 * c[i];
o1[i] = volnew_o8;
} 
}
int main()
{
double o1[101];
double c[101];
int i;
int len = 100;
#pragma omp parallel for simd
for (i = 0; i < len; ++i) {
c[i] = i + 1.01;
o1[i] = i + 1.01;
} 
foo1 (&o1[1], &o1[0], 100);
for (i = 0; i < len; ++i) {
printf("%lf\n",o1[i]);
}  
return 0;
}
