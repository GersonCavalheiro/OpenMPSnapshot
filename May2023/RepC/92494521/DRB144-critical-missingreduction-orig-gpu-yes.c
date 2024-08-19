#include <stdio.h>
#include <omp.h>
#define N 100
int var = 0;
int main(){
#pragma omp target map(tofrom:var) device(0)
#pragma omp teams distribute parallel for
for(int i=0; i<N*2; i++){
#pragma omp critical
var++;
}
printf("%d\n ",var);
return 0;
}
