

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include "hsaxpy.h"
#include "asaxpy.h"
#include "check1ns.h"
#include "wtcalc.h"

#define TWO26 (1 << 26)
#define NLUP  (32)


int main(int argc, char *argv[])
{
int    i, n,
iret,
ial;
size_t nbytes;
float  a = 2.0f,
*x, *y,
*yhost,
*yaccl,
maxabserr;
struct timespec rt[2];
double wt; 


check1ns();
printf("The system supports 1 ns time resolution\n");

if (0 == omp_get_num_devices()) {
printf("No accelerator found ... exit\n");
exit(EXIT_FAILURE);
}

n      = TWO26;
nbytes = sizeof(float) * n;
iret   = 0;
if (NULL == (x     = (float *) malloc(nbytes))) iret = -1;
if (NULL == (y     = (float *) malloc(nbytes))) iret = -1;
if (NULL == (yhost = (float *) malloc(nbytes))) iret = -1;
if (NULL == (yaccl = (float *) malloc(nbytes))) iret = -1;
if (0 != iret) {
printf("error: memory allocation\n");
free(x);     free(y);
free(yhost); free(yaccl);
exit(EXIT_FAILURE);
}
#pragma omp parallel for default(none) \
shared(a, x, y, yhost, yaccl, n) private(i)
for (i = 0; i < n; ++i) {
x[i]     = rand() % 32 / 32.0f;
y[i]     = rand() % 32 / 32.0f;
yhost[i] = a * x[i] + y[i]; 
yaccl[i] = 0.0f;
}
printf("total size of x and y is %9.1f MB\n", 2.0 * nbytes / (1 << 20));
printf("tests are averaged over %2d loops\n", NLUP);


memcpy(yaccl, y, nbytes);
wtcalc = -1.0;
hsaxpy(n, a, x, yaccl);
maxabserr = -1.0f;
for (i = 0; i < n; ++i) {
maxabserr = fabsf(yaccl[i] - yhost[i]) > maxabserr?
fabsf(yaccl[i] - yhost[i]) : maxabserr;
}
hsaxpy(n, a, x, yaccl);
wtcalc = 0.0;
clock_gettime(CLOCK_REALTIME, rt + 0);
for (int ilup = 0; ilup < 1; ++ilup) {
hsaxpy(n, a, x, yaccl);
}
clock_gettime(CLOCK_REALTIME, rt + 1);
wt=(rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
printf("saxpy on host: %9.1f MB/s %9.1f MB/s maxabserr = %9.1f\n",
3.0 * nbytes / ((1 << 20) * wt),
3.0 * nbytes / ((1 << 20) * wtcalc), maxabserr);


for (ial = 0; ial < 6; ++ial) {

memcpy(yaccl, y, nbytes);
wtcalc = -1.0;
asaxpy(n, a, x, yaccl, ial);
maxabserr = -1.0f;
for (i = 0; i < n; ++i) {
maxabserr = fabsf(yaccl[i] - yhost[i]) > maxabserr?
fabsf(yaccl[i] - yhost[i]) : maxabserr;
}
asaxpy(n, a, x, yaccl, ial);
wtcalc = 0.0;
clock_gettime(CLOCK_REALTIME, rt + 0);
for (int ilup = 0; ilup < NLUP; ++ilup) {
asaxpy(n, a, x, yaccl, ial);
}
clock_gettime(CLOCK_REALTIME, rt + 1);
wt=(rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
printf("saxpy on accl (impl. %d)\ntotal: %9.1f MB/s kernel: %9.1f MB/s maxabserr = %9.1f\n\n",
ial, NLUP * 3.0 * nbytes / ((1 << 20) * wt),
NLUP * 3.0 * nbytes / ((1 << 20) * wtcalc), maxabserr);
}

free(x);     free(y);
free(yhost); free(yaccl);
return 0;
}
