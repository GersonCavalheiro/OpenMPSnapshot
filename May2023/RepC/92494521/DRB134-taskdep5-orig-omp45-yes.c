#include <stdio.h>
#include <omp.h>
void foo(){
int x = 0, y = 2;
#pragma omp task depend(inout: x) shared(x)
x++;                                                             
#pragma omp task depend(in: x) depend(inout: y) shared(x, y)
y -= x;                                                         
#pragma omp task depend(in: x) if(0)                             
{}
printf("x=%d\n",x);
printf("y=%d\n",y);
#pragma omp taskwait		                                         
}
int main(){
#pragma omp parallel
#pragma omp single
foo();
return 0;
}
