#include <stdio.h>
#include <omp.h>
int main(){
const int N = 100;
int i, k, parts;
float a[N], b[N];
float dot_prod;
for(k = 0; k < N; k++){
a[k] = 3.0 * k;
b[k] = 1.8 * k;
}
#pragma omp parallel
{
#pragma omp for reduction(+:dot_prod)
for(i = 0; i < N; i++){
dot_prod  += a[i] * b[i];
}
}
printf("Inner product of a[] and b[] = %f\n", dot_prod);
return 0;
}	
