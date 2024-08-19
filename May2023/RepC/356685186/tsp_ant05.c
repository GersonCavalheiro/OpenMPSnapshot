#pragma GCC optimize("O3","unroll-loops","omit-frame-pointer","inline") 
#pragma GCC option("arch=native","tune=native","no-zero-upper") 
#pragma GCC target("avx")  
#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "stdbool.h"
#include "omp.h"
#define N  10000
#define Nx 1000
#define Ny 1000
#define nonExist -999999
#define ALPHA 0.50 
#define BETA  2.00 
#define RHO   0.50 
#define TAU_INITIAL_VALUE 0.50 
#define ANTS  100
#define REPETITIONS 8
#define DEBUG 0
#define THREADS 12
float CitiesX[N];
float CitiesY[N];
double CalculatedDistances_to_mBETA[N][N];
double TauValues_to_A[N][N]; 
double DistanceTravelled[ANTS]; 
int AntsPaths[ANTS][N+1]; 
double InvPathDepth[N][N]; 
void UpdatePathDepths() {
for (int i=0; i<N; i++) for (int j=0; j<N; j++) InvPathDepth[i][j] = 0;
for (int ant=0; ant<ANTS; ant++) {
for (int i=0; i<N+1; i++){
int city1 = AntsPaths[ant][i], city2 = AntsPaths[ant][i+1];
double temp = DistanceTravelled[ant];
InvPathDepth[city1][city2] = 1/temp;
InvPathDepth[city2][city1] = 1/temp;
}
}
}
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
double AvgOfDoubleArray(double ARRAY[], const int SIZE) {
double avg = 0.0;
for (int i=0; i<SIZE; i++) avg += ARRAY[i];
return avg/SIZE;
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
double Possibility(int ant, int I, int J, int ant_path_length, bool TheAntHasVisitiedCity[N]) {
double summation = 0.0;
for (int j=0; j<N; j++) 
if (!TheAntHasVisitiedCity[j]) 
summation += TauValues_to_A[I][j] * CalculatedDistances_to_mBETA[I][j];
return (TauValues_to_A[I][J] * CalculatedDistances_to_mBETA[I][J] / summation);
}
double CalculateNewTau_2(int I, int J) {
if (DEBUG==2) printf("Calculating new tau value between %d and %d...\n", I, J);
double DeltaTau = InvPathDepth[I][J];
return ( ((1-RHO) * pow(TauValues_to_A[I][J], 1.0/(float) ALPHA) ) + DeltaTau );
}
void CalculateNewTaus() {
printf("Now calculating new tau values of all pairs of cities...\n");
#pragma omp parallel for schedule(dynamic, 50) num_threads(THREADS)
for (int i=0; i<N; i++) {
for (int j=i+1; j<N; j++) {
double newTau = pow(CalculateNewTau_2(i, j), ALPHA);
TauValues_to_A[i][j] = newTau;
TauValues_to_A[j][i] = newTau;  
}
} 
}
void InitializeTauValues_2() {
printf("Now initializing the tau values...\n");
for (int i=0; i<N; i++) {
printf("\r> Progress: %.2f%%", 100*(i+1)/((float)N));
for (int j=0; j<N; j++) {
TauValues_to_A[i][j] = pow(TAU_INITIAL_VALUE, ALPHA); 
}
}
printf(" ===> Completed.\n");
}
void CalculateAllDistances_2() {
printf("Now calculating distances and hetas^(BETA) between all pairs of cities...\n");
for (int i=0; i<N; i++) {
printf("\r> Progress: %.2f%%", 100*(i+1)/((float)N));
for (int j=i+1; j<N; j++) {
double temp = Distance(i, j); double temp_to_mBETA = pow(temp, -BETA);
CalculatedDistances_to_mBETA[i][j] = temp_to_mBETA;
CalculatedDistances_to_mBETA[j][i] = temp_to_mBETA;     
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
void AntRun(int ant, int starting_city, int repetitions) {
if (DEBUG==1) printf(">> Ant #%d is now running...\n", ant);
double totDist = 0.0;
int visited_cities = 1, current_city = starting_city;
AntsPaths[ant][0] = starting_city; 	AntsPaths[ant][N] = starting_city;
bool TheAntHasVisitiedCity[N];  
for (int i=0; i<N; i++) TheAntHasVisitiedCity[i] = false; 
TheAntHasVisitiedCity[starting_city] = true;
do {
if (DEBUG==1) printf("\r>> Progress: %.2f%%", 100*(visited_cities+1)/((float) N) );
TheAntHasVisitiedCity[current_city] = true;
double highest_decision_value = 0.0;
int next_city = -1;
for (int i=0; i<N; i++) {
if (TheAntHasVisitiedCity[i]) continue; 
unsigned seed = 3*ant + 17*omp_get_thread_num() + 22*repetitions  + 1112*omp_get_wtime();
double random_number = ((float) rand_r(&seed) )/((float)RAND_MAX);
double decision_value = random_number * Possibility(ant, current_city, i, visited_cities, TheAntHasVisitiedCity);
if (decision_value > highest_decision_value) { 
next_city = i;
highest_decision_value = decision_value;
} 
}
AntsPaths[ant][visited_cities++] = next_city; 
totDist += Distance(current_city, next_city); 
current_city = next_city; 
} while (visited_cities < N);
totDist += Distance(current_city, starting_city);   
DistanceTravelled[ant] = totDist;
if (DEBUG==1) printf(" ===> Finished\n");
}
int main( int argc, const char* argv[] ) {
printf("------------------------------------------------------------------------------\n");
printf("This program searches for the optimal traveling distance between %d cities,\n", N);
printf("spanning in an area of X=(0,%d) and Y=(0,%d)\n", Nx, Ny);
printf("------------------------------------------------------------------------------\n");
srand(1046900);
SetCities();
CalculateAllDistances_2();
InitializeTauValues_2();
int repetitions = 0;
printf("\n~~~~ NOW RUNNING THE MAIN SEQUENCE ~~~~\n==================================================\n");
do {
printf("Now the ants are running...\n");
#pragma omp parallel for schedule(dynamic, 5) num_threads(THREADS)
for (int ant=0; ant<ANTS; ant++) {
#pragma omp critical
printf("ant...\n");
unsigned seed = ant + 83*omp_get_thread_num() + 1297*repetitions  + 11*omp_get_wtime();
int starting_city = (int)(  ((float) rand_r(&seed) )*(N-1)/((float)RAND_MAX)  );
if (starting_city<0 || starting_city>N-1) exit(1);
AntRun(ant, starting_city, repetitions);
}
CalculateNewTaus();
printf("REPETITION: %9d   AVERAGE_PATH_LENGTH: %8.3lf\n", ++repetitions, AvgOfDoubleArray(DistanceTravelled, ANTS));
printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
if (DEBUG==1) printf("\n");
} while (repetitions < REPETITIONS);
printf("\nCalculations completed. Results:\n");
printf("Optimal path distance found is: %.2lf\n", MinOfDoubleArray(DistanceTravelled, ANTS));
return 0 ;
}
