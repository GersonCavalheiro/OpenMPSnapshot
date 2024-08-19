#include <stdio.h>
#include <omp.h>
#define N 100
int var = 0;
int main(){
#pragma omp target map(tofrom:var) device(0)
#pragma omp teams distribute
for (int i=0; i<N; i++){
#pragma omp atomic update
var++;
}
printf("%d\n ",var);
return 0;
}
