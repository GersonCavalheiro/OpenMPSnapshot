#include <stdio.h>
#include <omp.h>
int main(){
int a, i;
#pragma omp parallel shared(a) private(i)
{
#pragma omp master
a = 0;
#pragma omp barrier
#pragma omp for reduction(+:a)
for (i=0; i<10; i++){
a += i;
}
#pragma omp single
printf("Sum is %d\n", a);
}
return 0;
}
