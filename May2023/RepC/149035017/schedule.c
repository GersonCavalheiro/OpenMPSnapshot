#define _BSD_SOURCE
#include <omp.h>
#include <stdio.h>
#include <unistd.h>
#define WORK 200
#define AVGTIME 50000
void work(int thread) {
int threads = omp_get_num_threads();
if (threads > 1) {
unsigned int work = (AVGTIME/2) + (AVGTIME / (omp_get_num_threads() - 1)) * thread;
usleep(work);
} else {
usleep(AVGTIME);
}
}
void print(int thread, int workitem) {	
for (int i = 0; i < thread; ++i) {
printf("\t");
}
printf("%d\n", workitem);
}
int main(int argc, char **argv) {
double t = omp_get_wtime();
#pragma omp parallel
{
int tid = omp_get_thread_num();
#pragma omp for schedule(guided, 1)
for (int i = 0; i < WORK; ++i) {
work(tid);
#pragma omp critical
{
print(tid, i);
}
}
}
t = omp_get_wtime() - t;
printf("%fs\n", t);
return 0;
}
