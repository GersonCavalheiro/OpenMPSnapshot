#include <stdio.h>
int main()
{
int i;
int s = 0;
int t = 0;
int d = 0;
float e = 0.0f;
float f = 0.0f;
#pragma omp simd for reduction(+:s,t) 
for(i=0; i<100; i++)
{
s += (i+1);
f += (i+1.0f);
}
#pragma omp simd for reduction(-:d, e) 
for(i=0; i<100; i++)
{
d -= (i+1);
e -= (i+1.0f);
}
printf("%d %f %d %f\n", s, f, d, e);
if ((s != 5050) || (f != 5050.0f)
|| (d != -5050) || (e != -5050.0f))
return 1;
return 0;
}
