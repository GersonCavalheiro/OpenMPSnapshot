#include<stdio.h>
#include<omp.h>
int main(int argc, char *argv[]){
int j, a[50], b[50], c[50], d[50], e[50], z[50], n = 50, f = 2;
omp_set_num_threads(4);
#pragma omp parallel for
for(j = 0; j < n; j++)
b[j] = c[j] = e[j] = j;
#pragma omp parallel
{
#pragma omp for nowait 
for(j=0;  j<n; j++){
a[j] = b[j] + c[j];
}    
#pragma omp for nowait
for(j=0; j<n; j++)
d[j] = e[j] *f;
#pragma omp barrier 
#pragma omp for
for(j=0; j<n; j++)
z[j] =  (a[j]+a[j+1])*0.5;
}
puts("Vetor A");
for(j=0;  j<50; j++)
printf("%d - ", a[j]);
puts("Vetor d");
for(j=0;  j<50; j++)
printf("%d - ", d[j]);
puts("Vetor z");
for(j=0;  j<50; j++)
printf("%d - ", z[j]);
return 0;
}