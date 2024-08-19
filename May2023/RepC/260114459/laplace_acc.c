#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#define GRIDY    2048
#define GRIDX    2048
#define MAX_TEMP_ERROR 0.02
double T_new[GRIDX+2][GRIDY+2]; 
double T[GRIDX+2][GRIDY+2];     
void init();
int main(int argc, char *argv[]) {
int i, j;                                            
int max_iterations;                                  
int iteration=1;                                     
double dt=100;                                       
struct timeval start_time, stop_time, elapsed_time;  
if(argc!=2) {
printf("Usage: %s number_of_iterations\n",argv[0]);
exit(1);
} else {
max_iterations=atoi(argv[1]);
}
gettimeofday(&start_time,NULL); 
init();                  
while ( dt > MAX_TEMP_ERROR && iteration <= max_iterations ) {
#pragma acc kernels
for(i = 1; i <= GRIDX; i++) 
for(j = 1; j <= GRIDY; j++) 
T_new[i][j] = 0.25 * (T[i+1][j] + T[i-1][j] +
T[i][j+1] + T[i][j-1]);
dt = 0.0;
#pragma acc kernels
for(i = 1; i <= GRIDX; i++){
for(j = 1; j <= GRIDY; j++){
dt = fmax( fabs(T_new[i][j]-T[i][j]), dt);
T[i][j] = T_new[i][j];
}
}
if((iteration % 100) == 0) 
printf("Iteration %4.0d, dt %f\n",iteration,dt);
iteration++;
}
gettimeofday(&stop_time,NULL);
timersub(&stop_time, &start_time, &elapsed_time); 
printf("Total time was %f seconds.\n", elapsed_time.tv_sec+elapsed_time.tv_usec/1000000.0);
return 0;
}
void init(){
int i,j;
for(i = 0; i <= GRIDX+1; i++){
for (j = 0; j <= GRIDY+1; j++){
T[i][j] = 0.0;
}
}
for(i = 0; i <= GRIDX+1; i++) {
T[i][0] = 0.0;
T[i][GRIDY+1] = (128.0/GRIDX)*i;
}
for(j = 0; j <= GRIDY+1; j++) {
T[0][j] = 0.0;
T[GRIDX+1][j] = (128.0/GRIDY)*j;
}
}
