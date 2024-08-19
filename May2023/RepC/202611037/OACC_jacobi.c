#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#define COLUMNS 1000
#define ROWS 1000
#define MAX_TEMP_ERROR 0.01
double Anew[ROWS+2][COLUMNS+2];   
double A[ROWS+2][COLUMNS+2]; 
void initialize();
void track_progress(int iter);
int main(int argc, char *argv[]) {
int i, j;                          
int max_iterations=1000;   
int iteration=1;       
double dt=100;
double t_final, t_inicial;                        
initialize();
t_inicial = omp_get_wtime();	
#pragma acc data create(Anew) copy(A)
{ 
while ( dt > MAX_TEMP_ERROR && iteration <= max_iterations ) {
#pragma acc parallel loop
for(i = 1; i <= ROWS; i++) {
for(j = 1; j <= COLUMNS; j++) {
Anew[i][j] = 0.25 * (A[i+1][j] + A[i-1][j] + A[i][j+1] + A[i][j-1]);
}
}
dt = 0.0;
#pragma acc parallel loop reduction(max:dt)
for(i = 1; i <= ROWS; i++){
for(j = 1; j <= COLUMNS; j++){
dt = fmax( fabs(Anew[i][j]-A[i][j]), dt);
A[i][j] = Anew[i][j];
}
}
iteration++;
}
}
printf("\nMax error at iteration %d was %f\n", iteration-1, dt);
t_final = omp_get_wtime();
printf("Tempo: %3.2f\n", t_final - t_inicial);
return 0;
}
void initialize(){
int i,j;
for(i = 0; i <= ROWS+1; i++){
for (j = 0; j <= COLUMNS+1; j++){
A[i][j] = 0.0;
}
}
for(i = 0; i <= ROWS+1; i++) {
A[i][0] = 0.0;
A[i][COLUMNS+1] = (100.0/ROWS)*i;
}
for(j = 0; j <= COLUMNS+1; j++) {
A[0][j] = 0.0;
A[ROWS+1][j] = (100.0/COLUMNS)*j;
}
}
