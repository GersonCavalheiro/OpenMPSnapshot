#include<omp.h>
#include<stdio.h>
#include<stdlib.h>
int main(int argc,char *argv){
int i,k,suma;
puts("--- Sin constructo Atomic ---");
for(k=0;k<5;k++){
suma = 0;
#pragma omp parallel for
for(i=0;i<500000;i++)
suma = suma + i;
printf("Suma total: %d\n",suma);
}
puts("--- Con constructo Atomic ---");
for(k=0;k<5;k++){
suma = 0;
#pragma omp parallel for
for(i=0;i<500000;i++)
#pragma omp atomic
suma = suma + i;
printf("Suma total: %d\n",suma);
}
return 0;
}
