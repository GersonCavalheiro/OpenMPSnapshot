#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
double findAvg(int rows, double* strip[rows], int cols)
{
int i, j;
double tot = 0.0;
for(j = 1; j < (cols-1); j++)
{
for(i = 1; i < (rows-1); i++)
{
tot = tot + strip[i][j];
}
}
tot = tot / ((rows-2)*(cols-2));
return tot;
}
int main(int argc, char* argv[])
{
int i, j, k, nt;
int rows, cols;
int count;
int tid;
rows = 1002;
cols = 30002;
double time1;
double* strip[rows];
double avg, lastavg, error;
double* lbuf[rows];
double* rbuf[rows];
for(i = 0; i < rows; i++)
{
strip[i] = (double*) malloc(cols*sizeof(double));
}
for(i = 0; i < rows; i++)
{
strip[i][0] = -50.0;
strip[i][(cols-1)] = -50.0;
for(j = 1; j < (cols-1); j++)
{
strip[i][j] = 0.0;
}
}
for(j = 1; j < (cols-1); j++)
{
strip[0][j] = 100.0;
strip[(rows-1)][j] = 100.0;
}
if(argc >= 2)
{
nt = atoi(argv[1]);
printf("\n\nNumber of Active Threads %02d\n", nt);
}
else
{
printf("Command Line : %s", argv[0]);
for(i = 1; i < argc; i++) printf("%s ", argv[i]);
printf("\n");
return 0;
}
for(i = 0; i < nt; i++)
{
lbuf[i] = (double*) malloc(rows*sizeof(double));
rbuf[i] = (double*) malloc(rows*sizeof(double));
}
for(i = 0; i < rows; i ++)
{
lbuf[0][i] = -50.0;
rbuf[(nt-1)][i] = -50.0;
}
omp_set_num_threads(nt);
omp_set_dynamic(0);
time1 = omp_get_wtime();
#pragma omp parallel
{
int tid, rc, lc, i, j, k;
tid = omp_get_thread_num();
if(tid < (nt-1)) rc = ((tid+1)*(cols-2))/nt;
else rc = (cols-1);
if(tid > 0) lc = ((tid*(cols-2))/nt) + 1;
else lc = 0;
avg = 0.0;
lastavg = -100.0;
count = 0;
error = fabs(avg - lastavg);
while( error > 0.005 )
{
#pragma omp for collapse(2)
for(j = 1; j < (cols-1); j++)
{
for(i = 1; i < (rows-1); i++)
{
if(j == rc)      rbuf[tid][i] = (strip[(i-1)][j] + strip[(i+1)][j] + strip[i][(j-1)] + strip[i][(j+1)])/4.0;
else if(j == lc) lbuf[tid][i] = (strip[(i-1)][j] + strip[(i+1)][j] + strip[i][(j-1)] + strip[i][(j+1)])/4.0;
else             strip[i][j]  = (strip[(i-1)][j] + strip[(i+1)][j] + strip[i][(j-1)] + strip[i][(j+1)])/4.0;
}
}
for(i = 1; i < (rows-1); i++)
{
strip[i][rc] = rbuf[tid][i];
strip[i][lc] = lbuf[tid][i];
}
#pragma omp master 
{
lastavg = avg;
avg = findAvg(rows, strip, cols);
count++;
error = fabs(avg - lastavg);
}
#pragma omp barrier 
}
}
time1 = omp_get_wtime() - time1;
printf("Average %f, count %02d\n", avg, count);
printf("Elapsed Time is %fs\n", time1);
return 0;
}
