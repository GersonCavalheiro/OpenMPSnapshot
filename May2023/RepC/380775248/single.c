#include <stdio.h>
#include <stdlib.h>
#include "omp.h"

int main(void) {

int x = 7, tid;  
#pragma omp parallel private(tid) shared(x) default(none)
{
tid = omp_get_thread_num();
printf("thread %d performed this task!\n", tid);
x++;

}

printf("x = %d\n", x);


}