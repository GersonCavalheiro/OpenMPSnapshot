#include <stdio.h>
#include <stdlib.h>
#include "omp.h"

int main(void) {

int tid = 50; 

#pragma omp parallel private(tid)
{
tid = omp_get_thread_num(); 
printf("Hello world from thread %d\n", tid);
}

puts("");
printf("tid = %d\n", tid);

}