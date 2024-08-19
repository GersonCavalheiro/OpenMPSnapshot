#include <omp.h>
#include <stdio.h>
static double x[20];
#pragma omp threadprivate(x)
int main(){
int i;
double j,k;
#pragma omp parallel for default(shared)
for (i = 0; i < 20; i++){
x[i] = -1.0;
if(omp_get_thread_num()==0){
j = x[0];
}
if(omp_get_thread_num()==0){
k = i+0.05;
}
}
printf ("%f %f\n", j, k);
return 0;
}
