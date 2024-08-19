#include <stdio.h>
#include <stdlib.h>
#include "omp.h"
#include "time.h"



int main(void) {

int x = 2, tid;
#pragma omp parallel private(tid) firstprivate(x)
{
x++;
tid = omp_get_thread_num();
printf("Thread %d: x = %d\n", tid, x);
}

printf("x = %d\n", x);


}