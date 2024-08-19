#pragma GCC optimize("O3","unroll-loops","omit-frame-pointer","inline") 
#pragma GCC option("arch=native","tune=native","no-zero-upper") 
#pragma GCC target("avx")  
#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#define N  10000
#define Nx 1000
#define Ny 1000
#define nonExist -999999
float CitiesX[N];
float CitiesY[N];
int Path[N+1];
double CalculatedDistances[N][N];
void SetCities() {
printf("Now initializing the positions of the cities...\n");
for (int i=0; i<N; i++) {
CitiesX[i] = Nx * (float) rand() / RAND_MAX;
CitiesY[i] = Ny * (float) rand() / RAND_MAX;
}
}
int IsInPath2(int city, int currentPathLength) {
for (int i=0; i<currentPathLength; i++)
if (Path[i] == city) return 1;
return 0;
}
void PrintCities() {
printf("> The cities are:\n");
for (int i=0; i<N; i++) {
printf(">> City: %6d  X:%5.2f Y:%5.2f\n", i, CitiesX[i], CitiesY[i] );
}
printf("\n");
}
void PrintPath() {
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
double PathDistance() {
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
double FindShortestStepPath() {
printf("Now finding the shortest step path...\n");
double totDist = 0.0;
int visited_cities = 1, current_city = 0;
Path[0] = 0; Path[N] = 0;
do {
printf("\r> Progress: %.2f%%", 100*(visited_cities+1)/((float)N));
double dist = 0, min_dist = INFINITY; 
int closest_city = -1;
for (int i=0; i<N; i++) {
if (IsInPath2(i, visited_cities)) continue; 
dist = CalculatedDistances[current_city][i];
if (min_dist > dist) {
min_dist = dist;
closest_city = i;
}
}
Path[visited_cities++] = closest_city;
totDist += min_dist;
current_city = closest_city;
} while (visited_cities<N);
printf(" ===> Completed.\n");
totDist += CalculatedDistances[Path[N-1]][0];
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
double totDistEstimation = FindShortestStepPath();
printf("\n");
printf("Estimated Total path distance is: %.2lf\n", totDistEstimation);
printf("Exact Total path distance is: %.2lf\n", PathDistance());
return 0 ;
}
