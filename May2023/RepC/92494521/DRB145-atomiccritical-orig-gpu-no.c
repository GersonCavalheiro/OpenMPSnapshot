#include <stdio.h>
#include <omp.h>
#define N 100
int var = 0;
int main(){
#pragma omp target map(tofrom:var) device(0)
#pragma omp teams distribute parallel for reduction(+:var)
for (int i=0; i<N; i++){
var++;
}
printf("%d\n",var);
return 0;
}
