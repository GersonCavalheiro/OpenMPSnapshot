#include <omp.h>
#include <stdio.h>
int main(){
int var=0,i;
#pragma omp target map(tofrom:var) device(0)
#pragma omp teams distribute parallel for
for (int i=0; i<100; i++){
var++;
}
printf("%d\n",var);
return 0;
}
