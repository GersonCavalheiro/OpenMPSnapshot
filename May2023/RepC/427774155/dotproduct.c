#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <sys/time.h>

#define N 100000000

double x[N], y[N];

double dotproduct_serial(int n, double x[], double y[]);
double dotproduct_omp_taskloop(int n, double x[], double y[]);

static void init_vectors(){

int i;
srand(time(NULL));
for(i = 0; i < N; i++){
x[i] = ((double) rand()/10.32)*((double) rand()/30.213);
}

for(i = 0; i < N; i++){
y[i] = ((double) rand())*((double) rand());
}
}

static void print_results(){

int i;
printf("x = [");
for (i = 0; i < N; i++){
printf("%lf ", x[i]);
}
printf("]\n");
printf("y = [");
for (i = 0; i < N; i++){
printf("%lf ", y[i]);
}
printf("]\n");
}

double dotproduct_serial(int n, double x[], double y[]){
int i;
double r = 0.0;

for(i = 0; i < n; i++){
r += x[i]*y[i];
}

return r;
}

double dotproduct_omp_taskloop(int n, double x[], double y[]){
int i;
double r = 0.0;

omp_set_num_threads(4);
#pragma omp parallel reduction(+: r)
{
#pragma omp single
#pragma omp taskloop num_tasks(10)
for(i = 0; i < n; i++){
r += x[i]*y[i];
}

}

return r;
}

int main(){

int i;
double exectime, exectimepar, res_serial, res_par;
struct timeval start, end;

if(N < 2){
fprintf(stderr, "Wrong Dimensions\n Exiting...\n");
exit(1);
}

printf("dot_product (x[%d], y[%d])\n", N, N);

init_vectors();

gettimeofday(&start, NULL);
res_serial = dotproduct_serial(N, x, y);
gettimeofday(&end, NULL);
exectime = (double) (end.tv_usec - start.tv_usec)*1E-06 + (double) (end.tv_sec - start.tv_sec); 


gettimeofday(&start, NULL);
res_par = dotproduct_serial(N, x, y);
gettimeofday(&end, NULL);
exectimepar = (double) (end.tv_usec - start.tv_usec)*1E-06 + (double) (end.tv_sec - start.tv_sec); 

printf("Execution time (serial):   %lf\n", exectime);
printf("Execution time (parallel): %lf\n", exectimepar);
printf("The dot product of the two vectors is result_s = %lf or result_p = %lf\n", res_serial, res_par);
return 0;
}
