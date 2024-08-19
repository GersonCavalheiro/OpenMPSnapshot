#pragma GCC optimize("O3","unroll-loops","omit-frame-pointer","inline") 
#pragma GCC option("arch=native","tune=native","no-zero-upper") 
#pragma GCC target("avx")  
#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "omp.h"
#define N 10000
#define Nx 1000
#define Ny 1000
#define VACANT_POSITION_CODE -999999
#define TOTAL_BATCHES 1e8
#define BATCH_SIZE 1200000
#define BATCH_SIZE_PER_RESCHEDULING 100000
#define DEFAULT_MAX_REPETITIONS TOTAL_BATCHES/BATCH_SIZE
float CitiesX[N];
float CitiesY[N];
int Path[N+1];
omp_lock_t Locks[N+1];
void SetCities() {
printf("Now initializing the positions of the cities...\n");
for (int i=0; i<N; i++) {
CitiesX[i] = Nx * (float) rand() / RAND_MAX;
CitiesY[i] = Ny * (float) rand() / RAND_MAX;
}
}
void ResetPath() {
printf("Now initializing the path...\n");
for (int i=0; i<N+1; i++)
Path[i] = -1;
}
int IsInPath(int k) {
for (int i=0; i<N; i++)
if (Path[i] == k) return 1;
return 0;
}
void RandomizePath() {
int k;
printf("Now randomizing the path...\n");
Path[0] = (N*rand())/RAND_MAX;
Path[N] = Path[0];
for (int i=1; i<N; i++) {
do {
k = ((float)N*rand())/RAND_MAX;
} while (IsInPath(k) == 1);
Path[i] = k;
}
}
void PrintCities() {
int x, y;
printf("> The cities are:\n");
for (int i=0; i<N; i++) {
printf(">> City: %6d  X:%5.2f Y:%5.2f\n", i, CitiesX[i], CitiesY[i] );
}
printf("\n");
}
void MapCities() {
int Map[Ny+1][Nx+1];
printf("Now creating a visual map of the cities...\n");
for (int i=0; i<Nx+1; i++) 
for (int j=0; j<Ny+1; j++) 
Map[j][i] = (float) VACANT_POSITION_CODE;
for (int c=0; c<N; c++) {
int x = (int) CitiesX[c] ;
int y = (int) CitiesY[c] ;
if (Map[y][x] == VACANT_POSITION_CODE) Map[y][x] = c+1;
else Map[y][x] = -1;
}
printf("This is the cities' map:\n");
printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
for (int y=0; y<Ny+1; y++){
for (int x=0; x<Nx+1; x++)
printf("%8d ", Map[y][x]);
printf("\n");
}
printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
printf("\n");
}
double Distance(int A, int B) {
double result = sqrt(   (CitiesX[A]-CitiesX[B])*(CitiesX[A]-CitiesX[B]) + (CitiesY[A]-CitiesY[B])*(CitiesY[A]-CitiesY[B]) );
return result;
}
double PathDistance() {
double totDist = 0.0;
for (int i=0; i<N; i++) {
totDist += Distance(Path[i], Path[i+1]);
}
totDist += Distance(Path[N], Path[0]);
return totDist;
}
double SwapCities(double totDist) {
double totDistChange = 0.0;
#pragma omp parallel for reduction(+:totDistChange) schedule(static, BATCH_SIZE_PER_RESCHEDULING) 
for (int counter=0; counter<BATCH_SIZE; counter++)
{
int A = (rand() %  (N-1 - 1 + 1)) + 1; 
int B = (rand() %  (N-1 - 1 + 1)) + 1; 
while (A==B) B = (rand() %  (N-1 - 1 + 1)) + 1; 
if (A>B) { int temp = A; A = B; B = temp; } 
int flag = B-A-1; 
double dist1_old, dist2_old, dist3_old, dist4_old, dist1_new=1, dist2_new, dist3_new, dist4_new;
dist1_old = Distance(Path[A-1], Path[A]); 
dist2_old = (!flag) ? 0 : Distance(Path[A], Path[A+1]); 
dist3_old = (!flag) ? 0 : Distance(Path[B-1], Path[B]); 
dist4_old = Distance(Path[B], Path[B+1]); 
dist1_new = Distance(Path[A-1], Path[B]); 
dist2_new = (!flag) ? 0 : Distance(Path[B], Path[A+1]); 
dist3_new = (!flag) ? 0 : Distance(Path[B-1], Path[A]); 
dist4_new = Distance(Path[A], Path[B+1]); 
double distChange = - dist1_old - dist2_old - dist3_old - dist4_old + dist1_new + dist2_new + dist3_new + dist4_new; 
if (distChange < 0) { 
omp_set_lock(&Locks[A]); omp_set_lock(&Locks[B]); 
int temp = Path[A];
Path[A] = Path[B];
Path[B] = temp;
omp_unset_lock(&Locks[A]); omp_unset_lock(&Locks[B]);
} else distChange=0;
totDistChange += distChange;
}
return totDist + totDistChange;
}
int ValidateParameters() {
if (Nx*Ny<N) return 0;
return 1;
}
void InitializeLocks() {
for (int i=0; i<N+1; i++)
omp_init_lock(&Locks[i]);
}
int main( int argc, const char* argv[] ) {
printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
printf("This program searches for the optimal traveling Distance between %d cities,\n", N);
printf("spanning in an area of X=(0,%d) and Y=(0,%d)\n", Nx, Ny);
printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
if (ValidateParameters() == 0) {
printf("\nERROR: NOT ENOUGH SPACE ALLOCATED FOR GIVEN NUMBER OF CITIES\n");
printf("The program will now exit.\n");
return 1;
}
int repetitions = 0, MaxRepetitions = DEFAULT_MAX_REPETITIONS;
if (argc>1) MaxRepetitions = atoi(argv[1]);
printf("Maximum number of repetitions set at: %d\n", MaxRepetitions);
printf("Maximum number of batches set at: %lf\n", TOTAL_BATCHES);
SetCities();
ResetPath();
RandomizePath();
InitializeLocks();
double totDist = PathDistance();
printf("Now running the main algorithm...\n");
do {
repetitions ++;
if (repetitions%10==0) printf(">>REPETITION:%8d  >>BATCH:%10d  >>PATH_LENGTH: %.1lf\n", repetitions, repetitions*BATCH_SIZE, totDist);	
totDist = SwapCities(totDist);
} while (repetitions < MaxRepetitions);
printf("\nCalculations completed. Results:\n");
printf("Repetitions: %d\n", repetitions);
printf("Batches: %d\n", repetitions*BATCH_SIZE);
printf("Actual optimal path length: %.2lf\n", PathDistance());
return 0 ;
}
