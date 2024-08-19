#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
void Hello(void);
int main(int argc, char* argv[])
{
int nthreads;                               
if(argv[1] == NULL){
printf("Invalid input. should run as ./program_name <num_of_threads>\n");
exit(0);
}
nthreads = strtol(argv[1], NULL, 10);
#pragma omp parallel num_threads(nthreads)
Hello();                                    
return 0;
}
void Hello(void){
int thread_rank = omp_get_thread_num();     
int thread_count = omp_get_num_threads();   
printf("Hello from thread %d of %d\n", thread_rank, thread_count);
}
