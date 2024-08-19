#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
int X_RESN = 2000; 
int Y_RESN = 2000; 
int MAX_ITER = 200;
struct timespec timer_start() {
struct timespec start_time;
clock_gettime(CLOCK_MONOTONIC, &start_time);
return start_time;
}
long long timer_end(struct timespec start_time){
struct timespec end_time;
clock_gettime(CLOCK_MONOTONIC, &end_time);
long long diffInNanos = (end_time.tv_sec - start_time.tv_sec) * (long long)1e9 + (end_time.tv_nsec - start_time.tv_nsec);
return diffInNanos;
}
typedef struct complextype {
double real, imag;
} Compl;
int main(int argc, char **argv)
{
int save = 0;
int s = 0;
for (int itA = 1; itA < argc; itA++) {
char *arg = argv[itA];
if (s == 0) {
if (strcmp(arg, "-s") == 0) save = 1;
if (strcmp(arg, "-x") == 0) s = 1;
if (strcmp(arg, "-y") == 0) s = 2;
if (strcmp(arg, "-i") == 0) s = 3;
if (strcmp(arg, "-xy") == 0) s = 4;
}
else 
{
if (s == 1) X_RESN = atoi(arg);
if (s == 2) Y_RESN = atoi(arg);
if (s == 3) MAX_ITER = atoi(arg);
if (s == 4) X_RESN = Y_RESN = atoi(arg);
s = 0;
}
}
if (s != 0) {
fprintf(stderr, "Missing argument value!");
exit(1);
}
struct timespec vartime = timer_start();
int *ks;
ks = (int *)malloc((X_RESN*Y_RESN) * sizeof(int));
double *ds;
ds = (double *)malloc((X_RESN*Y_RESN) * sizeof(double));
int i, j, k;
double d;
Compl z, c, t;
#pragma acc parallel loop copyout(ks[0:X_RESN*Y_RESN], ds[0:X_RESN*Y_RESN]) private(k, d, c, z, t)
for (int it = 0; it < X_RESN*Y_RESN; it++)
{
i = it / Y_RESN;
j = it % Y_RESN;
double u = ((double)i - (X_RESN / 2.0)) / (X_RESN / 4.0);
double v = ((double)j - (Y_RESN / 2.0)) / (Y_RESN / 4.0);
z.real = z.imag = 0.0;
c.real = v;
c.imag = u;
k = 0;
d = 0.0;
double lengthsq;
do
{
t = z;
z.imag = 2.0 * t.real * t.imag + c.imag;
z.real = t.real * t.real - t.imag * t.imag + c.real;
lengthsq = z.real * z.real + z.imag * z.imag;
d += pow(pow(z.imag - t.imag, 2.0) + pow(z.real - t.real, 2.0), 0.5);
k++;
} while (lengthsq < 4.0 && k < MAX_ITER);
ks[it] = k;
ds[it] = d;
}
free(ks);
free(ds);
long long time_elapsed_nanos = timer_end(vartime);
double elapsed = time_elapsed_nanos*0.000000001;
printf("%lf seconds elapsed\n", elapsed);
return 0;
}
