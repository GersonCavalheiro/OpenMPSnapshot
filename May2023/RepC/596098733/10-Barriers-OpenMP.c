#include <stdio.h>
#include <omp.h>
#include <unistd.h>
#define NUM_THREADS 4
int main(int argc, char* argv[]) {
int i, tid;
omp_set_num_threads(NUM_THREADS);
#pragma omp parallel private(i, tid)
{
tid = omp_get_thread_num();
for (i = 0; i < 5; i++) {
printf("Thread %d is performing iteration %d\n", tid, i);
if(tid%2==0){
sleep(1);
}else{
sleep(3);
}
}
#pragma omp barrier
printf("Thread %d finished its work\n", tid);
}
return 0;
}
