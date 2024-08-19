#pragma GCC optimize("O3","unroll-loops","omit-frame-pointer","inline") 
#pragma GCC option("arch=native","tune=native","no-zero-upper") 
#pragma GCC target("avx")  
#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "omp.h"
#include "stdbool.h"
#define N  10000
#define Nx 1000
#define Ny 1000
#define nonExist -999999
#define PICK_CLOSEST_CITY_POSSIBILITY 0.90
#define THREADS 12
float CitiesX[N];
float CitiesY[N];
int ThreadsPath[THREADS][N+1];
double CalculatedDistances[N][N];
void SetCities() {
printf("Now initializing the positions of the cities...\n");
for (int i=0; i<N; i++) {
CitiesX[i] = Nx * (float) rand() / RAND_MAX;
CitiesY[i] = Ny * (float) rand() / RAND_MAX;
}
}
void PrintCities() {
printf("> The cities are:\n");
for (int i=0; i<N; i++) {
printf(">> City: %6d  X:%5.2f Y:%5.2f\n", i, CitiesX[i], CitiesY[i] );
}
printf("\n");
}
void PrintPath_2(int Path[]) {
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
double PathDistance_2(int Path[]) {
double totDist = 0.0;
for (int i=0; i<N; i++) {
totDist += Distance(Path[i], Path[i+1]);
}
totDist += Distance(Path[N], Path[0]);
return totDist;
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
double FindShortestStepPath_2() {
#pragma omp master
{
printf("Now finding the shortest / second shortest step path...\n");
printf("> Threads running independently in parallel: %d\n", omp_get_num_threads());
}
double totDist = 0.0;
int visited_cities = 1, current_city = 0, thread = omp_get_thread_num();
bool CityIsVisited[N]; for (int i=0; i<N; i++) CityIsVisited[i] = false;
ThreadsPath[thread][0] = current_city; ThreadsPath[thread][N] = current_city; CityIsVisited[current_city] = false;
do {
#pragma omp master
printf("\r> Progress: %.2f%%", 100*(visited_cities)/((float)N));
double dist = 0, min_dist_1 = INFINITY, min_dist_2 = INFINITY; 
int closest_city_1 = -1, closest_city_2 = -1;
for (int i=0; i<N; i++) {
if (CityIsVisited[i] == true) continue; 
dist = CalculatedDistances[current_city][i];
if (min_dist_1 > dist) {
min_dist_2 = min_dist_1; closest_city_2 = closest_city_1;
min_dist_1 = dist; closest_city_1 = i;
} else if (min_dist_2 > dist) {
min_dist_2 = dist; closest_city_2 = i;
}
}
unsigned seed = 11*visited_cities + 83*thread + 11*omp_get_wtime() + current_city;
float random_number = ((float)rand_r(&seed)) / ((float)RAND_MAX) ;
int city_pick = (random_number<PICK_CLOSEST_CITY_POSSIBILITY) ? 1 : 2;
int next_city =  (city_pick==1) ? closest_city_1 : closest_city_2;
ThreadsPath[thread][visited_cities++] = next_city; 
CityIsVisited[next_city] = true;
current_city = next_city;
totDist += (city_pick==1) ? min_dist_1 : min_dist_2;; 
} while (visited_cities<N);
totDist += CalculatedDistances[ThreadsPath[thread][N-1]][0];
#pragma omp barrier
#pragma omp single
printf("\r> Progress: 100.00%% ===> Completed.\n");
#pragma omp barrier
return totDist;
}
int main( int argc, const char* argv[] ) {
printf("------------------------------------------------------------------------------\n");
printf("This program searches for the optimal traveling distance between %d cities,\n", N);
printf("spanning in an area of X=(0,%d) and Y=(0,%d)\n", Nx, Ny);
printf("------------------------------------------------------------------------------\n");
srand(1046900);
SetCities();
CalculateAllDistances();
double totDistEstimation = INFINITY;
#pragma omp parallel reduction(min:totDistEstimation) num_threads(THREADS)
{
totDistEstimation = FindShortestStepPath_2();
}
printf("\n");
printf("Minimum total path distance found is: %.2lf\n", totDistEstimation);
return 0 ;
}
