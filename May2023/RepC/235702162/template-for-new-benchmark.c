#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include "../polybench/polybench.h"
#include "template-for-new-benchmark.h"
#include <omp.h> 
static void init_array(int n,double C[1024][1024])
{
int i;
int j;
#pragma omp parallel for private (i,j) firstprivate (n)
for (i = 0; i <= n - 1; i += 1) {
#pragma omp parallel for private (j)
for (j = 0; j <= n - 1; j += 1) {
C[i][j] = 42;
}
}
}
static void print_array(int n,double C[1024][1024])
{
int i;
int j;
for (i = 0; i <= n - 1; i += 1) {
for (j = 0; j <= n - 1; j += 1) {
fprintf(stderr,"%0.2lf ",C[i][j]);
if (i % 20 == 0) 
fprintf(stderr,"\n");
}
}
fprintf(stderr,"\n");
}
static void kernel_template(int n,double C[1024][1024])
{
int i;
int j;
#pragma omp parallel for private (i,j) firstprivate (n)
for (i = 0; i <= n - 1; i += 1) {
#pragma omp parallel for private (j)
for (j = 0; j <= n - 1; j += 1) {
C[i][j] += 42;
}
}
}
int main(int argc,char **argv)
{
int n = 1024;
double (*C)[1024][1024];
C = ((double (*)[1024][1024])(polybench_alloc_data(((1024 + 0) * (1024 + 0)),(sizeof(double )))));
;
init_array(n, *C);
;
kernel_template(n, *C);
;
;
if (argc > 42 && !strcmp(argv[0],"")) 
print_array(n, *C);
free((void *)C);
;
return 0;
}
