#include<stdio.h>
#include<omp.h>
int main(int argc, char *argv[]){
int j, a[50], b[50], c[50];
omp_set_num_threads(4);
#pragma omp parallel for num_threads(4)
for(j = 0; j < 50; j++)
b[j] = c[j] = j;
#pragma omp parallel for num_threads(4) schedule(guided, 5)
for(j=0;  j<50; j++){
printf("J = %d, processo = %d\n", j, omp_get_thread_num());
a[j] = b[j] + c[j];
}    
for(j=0;  j<50; j++)
printf("%d - ", a[j]);
return 0;
}