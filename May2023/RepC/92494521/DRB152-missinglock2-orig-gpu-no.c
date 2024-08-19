#include <stdio.h>
#include <omp.h>
#define N 100
int main(){
omp_lock_t lck;
int var=0;
omp_init_lock(&lck);
#pragma omp target map(tofrom:var) device(0)
#pragma omp teams num_teams(1)
#pragma omp distribute parallel for
for (int i=0; i<N; i++){
omp_set_lock(&lck);
var++;
omp_unset_lock(&lck);
}
omp_destroy_lock(&lck);
printf("%d\n",var);
return 0;
}
