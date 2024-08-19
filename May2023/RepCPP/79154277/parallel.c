

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>

#define COLUMNS    1000
#define ROWS       1000

#define MAX_TEMP_ERROR 0.01

double Temperature[ROWS+2][COLUMNS+2];      
double Temperature_last[ROWS+2][COLUMNS+2]; 

void initialize();
void track_progress(int iter);


int main(int argc, char *argv[]) {

int i, j;                                            
int max_iterations = 4000;                                  
int iteration=1;                                     
double dt=100;                                       
struct timeval start_time, stop_time, elapsed_time;  


gettimeofday(&start_time,NULL); 

initialize();                   

#pragma acc data copy(Temperature_last) create(Temperature)
while ( dt > MAX_TEMP_ERROR && iteration <= max_iterations ) {

#pragma acc kernels
{
for(i = 1; i <= ROWS; i++) {
for(j = 1; j <= COLUMNS; j++) {
Temperature[i][j] = 0.25 * (Temperature_last[i+1][j] + Temperature_last[i-1][j] +
Temperature_last[i][j+1] + Temperature_last[i][j-1]);
}
}

dt = 0.0; 

for(i = 1; i <= ROWS; i++){
for(j = 1; j <= COLUMNS; j++){
dt = fmax( fabs(Temperature[i][j]-Temperature_last[i][j]), dt);
Temperature_last[i][j] = Temperature[i][j];
}
}
}
iteration++;
}

gettimeofday(&stop_time,NULL);
timersub(&stop_time, &start_time, &elapsed_time); 

printf("\nMax error at iteration %d was %f\n", iteration-1, dt);
printf("Total time was %f seconds.\n", elapsed_time.tv_sec+elapsed_time.tv_usec/1000000.0);

}


void initialize(){

int i,j;

for(i = 0; i <= ROWS+1; i++){
for (j = 0; j <= COLUMNS+1; j++){
Temperature_last[i][j] = 0.0;
}
}


for(i = 0; i <= ROWS+1; i++) {
Temperature_last[i][0] = 0.0;
Temperature_last[i][COLUMNS+1] = (100.0/ROWS)*i;
}

for(j = 0; j <= COLUMNS+1; j++) {
Temperature_last[0][j] = 0.0;
Temperature_last[ROWS+1][j] = (100.0/COLUMNS)*j;
}
}


void track_progress(int iteration) {

int i;

printf("---------- Iteration number: %d ------------\n", iteration);
for(i = ROWS-5; i <= ROWS; i++) {
printf("[%d,%d]: %5.2f  ", i, i, Temperature[i][i]);
}
printf("\n");
}
