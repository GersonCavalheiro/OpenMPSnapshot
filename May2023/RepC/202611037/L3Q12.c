#include<stdio.h>
#include<omp.h>
int main(int argc, char *argv[]){
int i = 23, x, n = 10, y[10];
omp_set_num_threads(3);
x= 1;
#pragma omp parallel for firstprivate (x)
for(i = 0; i < n; i++){
printf("Eu %d, com i = %d e x = %d\n", omp_get_thread_num(), i, x);
y[i] = x + i;
x = i;
}
for(i=0; i<n; i++)
printf("%d\n", y[i]);
return 0;
}