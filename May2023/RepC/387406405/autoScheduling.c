#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
void schedule_auto()
{
int N = 20;
int i;
double avg = 0;
omp_set_num_threads(4);
#pragma omp parallel for reduction(+:avg) schedule(auto)
for (i = 0; i < N; ++i) 
{
avg += i;
printf("Thread %d executing iteration %d\n",omp_get_thread_num(),i);
}
avg /= N;
printf("average is %.4f\n",avg);
}
int main()
{
printf("I'm from auto scheduling\n\n");
schedule_auto();
return 0;
} 
