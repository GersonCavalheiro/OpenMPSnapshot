#include <stdio.h>
#include <omp.h>
int main(){
int a, b, c, d;
#pragma omp parallel
#pragma omp single
{
#pragma omp task depend(out: c)
c = 1;
#pragma omp task depend(out: a)
a = 2;
#pragma omp task depend(out: b)
b = 3;
#pragma omp task depend(in: a)
c += a;
#pragma omp task depend(in: b)
c += b;
#pragma omp task depend(in: c)
d = c;
}
printf("%d\n",d);
return 0;
}
