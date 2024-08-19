#include <stdio.h>
#include <omp.h>
#define N 100
int main(){
int var = 0;
#pragma omp target map(tofrom:var) device(0)
#pragma omp teams num_teams(1)
#pragma omp distribute parallel for
for (int i=0; i<N; i++){
var++;
}
printf("%d\n ",var);
return 0;
}
