#include<omp.h>
#include<stdio.h>
#include<stdlib.h>
int main(int argc,char *argv){
int a = 10;
puts("------- Sección Paralela -------");
#pragma omp parallel firstprivate(a)
{
a += omp_get_thread_num();
printf("a = %d\n",a);
}
puts("------- Sección Serial -------");
printf("a = %d\n",a);
return 0;
}
