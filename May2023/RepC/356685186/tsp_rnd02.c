#pragma GCC optimize("O3","unroll-loops","omit-frame-pointer","inline") 
#pragma GCC option("arch=native","tune=native","no-zero-upper") 
#pragma GCC target("avx")  
#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#define N 10000
#define Nx 1000
#define Ny 1000
#define VACANT_POSITION_CODE -999999
#define DEFAULT_MAX_REPETITIONS 1e8
float CitiesX[N];
float CitiesY[N];
int Path[N+1];
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
return (double) sqrt(   (CitiesX[A]-CitiesX[B])*(CitiesX[A]-CitiesX[B]) + (CitiesY[A]-CitiesY[B])*(CitiesY[A]-CitiesY[B])   );
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
int A = (rand() %  (N-1 - 1 + 1)) + 1; 
int B = (rand() %  (N-1 - 1 + 1)) + 1; 
while (A==B) B = (rand() %  (N-1 - 1 + 1)) + 1;
if (A==0) while(1) printf("ERROR");if (B==0) while(1) printf("ERROR");if (A==N+1) while(1) printf("ERROR");if (B==N+1) while(1) printf("ERROR");
if (A>B) { int temp = A; A = B; B = temp; } 
int flag = B-A-1; 
double dist1_old = Distance(Path[A-1], Path[A]); 
double dist2_old = (!flag) ? 0 : Distance(Path[A], Path[A+1]); 
double dist3_old = (!flag) ? 0 : Distance(Path[B-1], Path[B]); 
double dist4_old = Distance(Path[B], Path[B+1]); 
double dist1_new = Distance(Path[A-1], Path[B]); 
double dist2_new = (!flag) ? 0 : Distance(Path[B], Path[A+1]); 
double dist3_new = (!flag) ? 0 : Distance(Path[B-1], Path[A]); 
double dist4_new = Distance(Path[A], Path[B+1]); 
double newDist = totDist - dist1_old - dist2_old - dist3_old - dist4_old + dist1_new + dist2_new + dist3_new + dist4_new;
if (newDist < totDist) {
int temp = Path[A];
Path[A] = Path[B];
Path[B] = temp;
double newDist1 = PathDistance();
if ((newDist1 - newDist >0.1)  || (newDist1 - newDist < -0.1) ) {
printf("ERROR NEW DIST IS NON-CONSISTENT WITH ITSELF. DIFF IS: %.20lf\n", newDist-newDist1);
printf("Tried to swap i=%d and j=%d  ||  path_i=%d and path_j=%d...\n",A,B,Path[A],Path[B]);
char c;
scanf("%c",&c);
} 
return newDist;
}
return totDist;
}
int ValidateParameters() {
if (Nx*Ny<N) return 0;
return 1;
}
int main( int argc, const char* argv[] ) {
printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
printf("This program searches for the optimal traveling distance between %d cities,\n", N);
printf("spanning in an area of X=(0,%d) and Y=(0,%d)\n", Nx, Ny);
printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
if (ValidateParameters() == 0) {
printf("\nERROR: NOT ENOUGH SPACE ALLOCATED FOR GIVEN NUMBER OF CITIES\n");
printf("The program will now exit.\n");
return 1;
}
int repetitions = 0;
int MaxRepetitions = DEFAULT_MAX_REPETITIONS;
if (argc>1) MaxRepetitions = atoi(argv[1]);
printf("Maximum number of repetitions set at: %d\n", MaxRepetitions);
SetCities();
ResetPath();
RandomizePath();
double totDist = PathDistance();
printf("Now running the main algorithm...\n");
do {
repetitions ++;
totDist = SwapCities(totDist);
if (repetitions%1000==0) printf("REPETITION: %9d   PATH_LENGTH: %8.3lf\n", repetitions, totDist);	
} while (repetitions < MaxRepetitions);
printf("\nCalculations completed. Results:\n");
printf("Repetitions: %d\n", repetitions);
printf("Estimation of optimal path length: %.2lf\n", totDist);
printf("Actual optimal path length: %.2lf\n", PathDistance());
return 0 ;
}
