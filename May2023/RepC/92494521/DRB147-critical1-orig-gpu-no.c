#include <omp.h>
#include <stdio.h>
#define N 100
int var = 0;
int main(){
#pragma omp target map(tofrom:var) device(0)
#pragma omp teams distribute parallel for
for(int i=0; i<N; i++){
#pragma omp atomic
var++;
#pragma omp atomic
var -= 2;
}
printf("%d\n",var);
return 0;
}
