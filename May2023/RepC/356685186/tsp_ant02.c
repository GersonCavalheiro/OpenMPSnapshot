#pragma GCC optimize("O3","unroll-loops","omit-frame-pointer","inline") 
#pragma GCC option("arch=native","tune=native","no-zero-upper") 
#pragma GCC target("avx")  
#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "stdbool.h"
#define N  50
#define Nx 1000
#define Ny 1000
#define nonExist -999999
#define ALPHA 0.50 
#define BETA  0.10 
#define GAMMA 1.00
#define RHO   0.50 
#define TAU_INITIAL_VALUE 0.50 
#define ANTS  100
#define REPETITIONS 20
#define DEBUG 0
float CitiesX[N];
float CitiesY[N];
double CalculatedDistances[N][N];
double TauValues_to_ALPHA[N][N]; 
double DistanceTravelled[ANTS]; 
int AntsPaths[ANTS][N+1]; 
void PrintIntArray(int ARRAY[], const int SIZE) {
for (int i=0; i<SIZE; i++) {
printf("%3d  ", ARRAY[i]);
}
printf("\n");
}
double MinOfDoubleArray(double ARRAY[], const int SIZE) {
double min = INFINITY;
for (int i=0; i<SIZE; i++)
if (ARRAY[i] < min) 
min = ARRAY[i];
return min;
}
void PrintCities() {
printf("> The cities are:\n");
for (int i=0; i<N; i++) {
printf(">> City: %6d  X:%5.2f Y:%5.2f\n", i, CitiesX[i], CitiesY[i] );
}
printf("\n");
}
void PrintPath_2(int Path[N+1]) {
printf("> The path is:\n");
for (int i=0; i<N+1; i++) {
printf(">> %d ", Path[i]);
}
printf("\n");
}
void MapCities() {
int Map[Ny+1][Nx+1];
printf("Now creating a visual map of the cities...\n");
for (int i=0; i<Nx+1; i++) 
for (int j=0; j<Ny+1; j++) 
Map[j][i] = (float) nonExist;
for (int c=0; c<N; c++) {
int x = (int) CitiesX[c] ;
int y = (int) CitiesY[c] ;
if (Map[y][x] == nonExist) Map[y][x] = c;
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
double PathDistance_2(int Path[N+1]) {
double totDist = 0.0;
for (int i=0; i<N; i++) {
totDist += Distance(Path[i], Path[i+1]);
}
totDist += Distance(Path[N], Path[0]);
return totDist;
}
double Possibility(int ant, int I, int J, int ant_path_length, bool TheAntHasVisitiedCity[N]) {
double summation = 0.0;
for (int j=0; j<N; j++) 
if (!TheAntHasVisitiedCity[j]) 
summation += TauValues_to_ALPHA[I][j] * pow(CalculatedDistances[I][j], -BETA);
if (isnan(summation)) {printf("\nFATAL ERROR: <summation> IS NAN\nTHE PROGRAM WILL BE TERMINATED.\n"); exit(1);}
return (TauValues_to_ALPHA[I][J] * pow(CalculatedDistances[I][J],-BETA) / summation);
}
double CalculateNewTau(int I, int J) {
if (DEBUG>1) printf("Calculating new tau value between %d and %d...\n", I, J);
double DeltaTau = 0.0;
for (int ant=0; ant<ANTS; ant++) {
if (DEBUG>1) printf("\r> Progress: %.2f%%", 100*(ant+1)/((float)ANTS));
for (int city=0; city<N; city++) {
if ( ((AntsPaths[ant][city]==I)||(AntsPaths[ant][city]==J)) && ((AntsPaths[ant][city+1]==I)||(AntsPaths[ant][city+1]==J)) ) {
DeltaTau += 1.0 / DistanceTravelled[ant];
break; 
}
}
} 
if (DEBUG==1) printf("Delta tau is: %.10lf\n", DeltaTau);
if (DEBUG>1) printf(" ===> Completed.\n");
return ( ((1-RHO) * pow(TauValues_to_ALPHA[I][J], 1.0/(float) ALPHA) ) + (GAMMA*DeltaTau) );
}
void CalculateNewTaus() {
if (DEBUG) printf("Now calculating new tau values of all pairs of cities...\n");
for (int i=0; i<N; i++) {
if (DEBUG) printf("\r> Progress: %.2f%%", 100*(i+1)/((float)N));
for (int j=i+1; j<N; j++) {
double newTau = pow(CalculateNewTau(i, j), ALPHA);
TauValues_to_ALPHA[i][j] = newTau;
TauValues_to_ALPHA[j][i] = newTau;  
}
} 
if (DEBUG) printf(" ===> Completed.\n");
}
void InitializeTauValues_2() {
printf("Now initializing the tau values...\n");
for (int i=0; i<N; i++) {
printf("\r> Progress: %.2f%%", 100*(i+1)/((float)N));
for (int j=0; j<N; j++) {
TauValues_to_ALPHA[i][j] = pow(TAU_INITIAL_VALUE, ALPHA); 
}
}
printf(" ===> Completed.\n");
}
void CalculateAllDistances() {
printf("Now calculating distances between all pairs of cities...\n");
for (int i=0; i<N; i++) {
printf("\r> Progress: %.2f%%", 100*(i+1)/((float)N));
for (int j=i+1; j<N; j++) {
double temp = Distance(i, j);
CalculatedDistances[i][j] = temp;
CalculatedDistances[j][i] = temp;        
}
}
printf(" ===> Completed.\n");
}
void SetCities() {
printf("Now initializing the positions of the cities...\n");
for (int i=0; i<N; i++) {
CitiesX[i] = Nx * (float) rand() / RAND_MAX;
CitiesY[i] = Ny * (float) rand() / RAND_MAX;
}
}
void AntRun(int ant, int starting_city) {
if (DEBUG) printf("> ANT %d IS RUNNING...\n", ant);
double totDist = 0.0;
int visited_cities = 1, current_city = starting_city;
AntsPaths[ant][0] = starting_city; 	AntsPaths[ant][N] = starting_city;
bool TheAntHasVisitiedCity[N]; for (int i=0; i<N; i++) TheAntHasVisitiedCity[i] = false;TheAntHasVisitiedCity[starting_city] = true;
do {
if (DEBUG) printf("\r>> Progress: %.2f%%", 100*(visited_cities+1)/((float) N) );
TheAntHasVisitiedCity[current_city] = true;
double highest_decision_value = 0.0;
int next_city = -1;
for (int i=0; i<N; i++) {
if (TheAntHasVisitiedCity[i]) continue; 
double random_number = 100000.0 * rand() / ((double) RAND_MAX); 
double decision_value = random_number * Possibility(ant, current_city, i, visited_cities, TheAntHasVisitiedCity);
if (DEBUG>1) printf(">>> test-city %d, random number: %.5lf, decision value: %.20lf\n", i, random_number, decision_value);
if (decision_value > highest_decision_value) { 
next_city = i;
highest_decision_value = decision_value;
} 
}
AntsPaths[ant][visited_cities++] = next_city; 
totDist += CalculatedDistances[current_city][next_city]; 
current_city = next_city; 
} while (visited_cities < N);
totDist += CalculatedDistances[current_city][starting_city];
DistanceTravelled[ant] = totDist;
if (DEBUG) printf(" ===> Finished\n");
}
int main( int argc, const char* argv[] ) {
printf("------------------------------------------------------------------------------\n");
printf("This program searches for the optimal traveling distance between %d cities,\n", N);
printf("spanning in an area of X=(0,%d) and Y=(0,%d)\n", Nx, Ny);
printf("------------------------------------------------------------------------------\n");
srand(1046900);
SetCities();
CalculateAllDistances();
InitializeTauValues_2();
int repetitions = 0;
do {
for (int ant=0; ant<ANTS; ant++) {
int starting_city = (int) ((double)N*rand()/((double) RAND_MAX) );
AntRun(ant, starting_city);
if (DEBUG>1) printf(">> Ants path was:\n>>> ");
if (DEBUG>1) PrintIntArray(AntsPaths[ant], N+1);
}
CalculateNewTaus();
printf("REPETITION: %9d   ESTIMATED_OPTIMAL_PATH_LENGTH: %8.3lf\n", ++repetitions, MinOfDoubleArray(DistanceTravelled, ANTS));
} while (repetitions < REPETITIONS);
printf("\nCalculations completed. Results:\n");
printf("Optimal path distance found is: %.2lf\n", MinOfDoubleArray(DistanceTravelled, ANTS));
return 0 ;
}
