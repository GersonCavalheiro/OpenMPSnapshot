#include <omp.h>
#include <stdio.h>
void foo() {
int a = 0;
#pragma omp parallel
#pragma omp single
{
#pragma omp task depend(inout : a) shared(a)
{
#pragma omp task depend(inout : a) shared(a)
a++;
}
#pragma omp task depend(inout : a) shared(a)
{
#pragma omp task depend(inout : a) shared(a)
a++;
}
}
printf("a=%d\n", a);
}
int main() {
foo();
return 0;
}
