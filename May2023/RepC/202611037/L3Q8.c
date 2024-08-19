#include<stdio.h>
#include<omp.h>
int main(int argc, char *argv[]){
int i, n = 50, a = 2, z[50], x[50], y = 5;
for(i=0;  i<50; i++)
x[i] = 3;
#pragma omp parallel num_threads(5) private(i)
for (i = omp_get_thread_num(); i < n; i+=omp_get_num_threads()){
z[i] = a * x[i] + y;
}
puts("Vetor z");
for(i=0;  i<50; i++)
printf("%d - ", z[i]);
return 0;
}