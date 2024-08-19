#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <omp.h>

double timeGetTime()
{
struct timeval time;
struct timezone zone;
gettimeofday(&time, &zone);
return time.tv_sec + time.tv_usec * 1e-6;
}

const long int VERYBIG = 100000;

int main(void)
{
int i;
long int j, k, sum;
double sumx, sumy, total, z;
double starttime, elapsedtime;
double thread_start_times[4];
double thread_end_times[4];
printf("Hossein Soltanloo - 810195407\n\n");
printf("OpenMP Parallel Timings for %ld iterations \n\n", VERYBIG);

for (i = 0; i < 6; i++)
{
starttime = timeGetTime();
sum = 0;
total = 0.0;

#pragma omp parallel num_threads(4) private(sumx, sumy, k)
{
thread_start_times[omp_get_thread_num()] = timeGetTime();
#pragma omp for reduction(+ \
: sum, total) schedule(runtime) nowait
for (int j = 0; j < VERYBIG; j++)
{
sum += 1;

sumx = 0.0;
for (k = 0; k < j; k++)
sumx = sumx + (double)k;

sumy = 0.0;
for (k = j; k > 0; k--)
sumy = sumy + (double)k;

if (sumx > 0.0)
total = total + 1.0 / sqrt(sumx);
if (sumy > 0.0)
total = total + 1.0 / sqrt(sumy);
}
thread_end_times[omp_get_thread_num()] = timeGetTime();
}

elapsedtime = timeGetTime() - starttime;

printf("Time Elapsed %10d mSecs Total=%lf Check Sum = %ld\n",
(int)(elapsedtime * 1000), total, sum);
for (size_t i = 0; i < 4; i++)
{
printf("T%d Time Elapsed %10d mSecs\n",
i, (int)((thread_end_times[i] - thread_start_times[i]) * 1000));
}
}

return 0;
}
