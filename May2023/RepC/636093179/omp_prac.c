#include <stdio.h>
#include <omp.h>
int main(){
int i;
int id;
#pragma omp parallel
{
id = omp_get_thread_num();
for( int i = 0; i < omp_get_max_threads(); i++){
if(i == omp_get_thread_num()){
printf("Hello from thread %d\n", id);
}
#pragma omp barrier
}
}
return 0;
}