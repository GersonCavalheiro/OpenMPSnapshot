

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

#define NC    1000
#define NR    1000

double t[NR+2][NC+2];      
double t_old[NR+2][NC+2];  

void initialize();
void set_bcs();
void print_trace(int iter);


int main(int argc, char *argv[]) {

double dt;                 
int i, j;                 
int niter;                
int iter;                 

printf("How many iterations [100-1000]? ");
scanf("%d", &niter);

initialize();            
set_bcs();               

for(i = 0; i <= NR+1; i++){
for(j=0; j<=NC+1; j++){
t_old[i][j] = t[i][j];
}
}


double start_time = omp_get_wtime();

#pragma omp parallel shared(t, t_old) private(i,j,iter) firstprivate(niter) reduction(max:dt)
for(iter = 1; iter <= niter; iter++) {

#pragma omp for
for(i = 1; i <= NR; i++) {
for(j = 1; j <= NC; j++) {
t[i][j] = 0.25 * (t_old[i+1][j] + t_old[i-1][j] +
t_old[i][j+1] + t_old[i][j-1]);
}
}

dt = 0.0;

#pragma omp for
for(i = 1; i <= NR; i++){
for(j = 1; j <= NC; j++){
dt = fmax( fabs(t[i][j]-t_old[i][j]), dt);
t_old[i][j] = t[i][j];
}
}

#pragma omp master
if((iter % 100) == 0) {
print_trace(iter);
}
#pragma omp barrier
}

double end_time = omp_get_wtime();

printf("Time taken: %f\n", end_time - start_time);
}


void initialize(){

int i,j;

for(i = 0; i <= NR+1; i++){
for (j = 0; j <= NC+1; j++){
t[i][j] = 0.0;
}
}
}


void set_bcs(){

int i,j;

for(i = 0; i <= NR+1; i++) {
t[i][0] = 0.0;
t[i][NC+1] = (100.0/NR)*i;
}

for(j = 0; j <= NC+1; j++) {
t[0][j] = 0.0;
t[NR+1][j] = (100.0/NR)*j;
}
}


void print_trace(int iter) {

int i;

printf("---------- Iteration number: %d ------------\n", iter);
for(i = NR-5; i <= NR; i++) {
printf("[%d,%d]: %5.2f  ", i, i, t[i][i]);
}
printf("\n");
}
