#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define N_POINTS 10000              
#define N_AGENTS 8                  
#define P 0.5                       
#define PHEROMONE_INIT_VAL (float)1 
struct AntAgent {
float pathLength;         
int city_flags[N_POINTS]; 
int route[N_POINTS];
int initialCity;
int currentCity;
};
float cities[N_POINTS][2] = {0};
float minPathLength = 0;
float avgPathLength = 0;
struct AntAgent ants[N_AGENTS];
float pheromones[N_POINTS][N_POINTS];
unsigned int seed = 159852753;
unsigned int randUint() {
#pragma omp threadprivate(seed)
seed = seed * 1103515245 + 12345;
return seed;
}
void initVec() {
for (int i = 0; i < N_POINTS; i++) {
cities[i][0] = (float)rand() / RAND_MAX * 1e3;
cities[i][1] = (float)rand() / RAND_MAX * 1e3;
}
}
void initPheromones() {
#pragma omp parallel for
for (int i = 0; i < N_POINTS; i++) {
for (int j = 0; j < N_POINTS; j++) {
pheromones[i][j] = PHEROMONE_INIT_VAL;
}
}
}
void resetAgents() {
for (int i = 0; i < N_AGENTS; i++) {
ants[i].pathLength = 0;
memset(&ants[i].city_flags[0], 1, N_POINTS * sizeof(int));
int register tmp = (int)rand() % N_POINTS;
ants[i].initialCity = tmp;
ants[i].currentCity = tmp;
ants[i].route[0] = tmp;
ants[i].city_flags[tmp] = 0;
}
}
float dist(int p1, int p2) {
float register dx = cities[p1][0] - cities[p2][0];
float register dy = cities[p1][1] - cities[p2][1];
return (float)sqrt(dx * dx + dy * dy);
}
void releaseAgents() {
#pragma omp parallel for
for (int i = 0; i < N_AGENTS; i++) {
float city_probs[N_POINTS] = {0};
for (int j = 0; j < N_POINTS - 1; j++) {
int register curr = ants[i].currentCity;
float prob = (float)randUint() / __UINT32_MAX__;
float denominator = 0;
for (int k = 0; k < N_POINTS; k++) {
if (ants[i].city_flags[k]) {
float len = dist(curr, k);
len = 1.0 / len;
float register tmp = sqrt(pheromones[curr][k] * len);
city_probs[k] = tmp;
denominator += tmp;
}
}
prob *= denominator;
float cumulativeProb = 0;
for (int k = 0; k < N_POINTS; k++) {
if (ants[i].city_flags[k]) {
cumulativeProb += city_probs[k];
if (prob < cumulativeProb) {
ants[i].city_flags[k] = 0;
ants[i].pathLength += dist(curr, k);
ants[i].currentCity = k;
ants[i].route[j + 1] = k;
break;
}
}
}
}
ants[i].pathLength += dist(ants[i].currentCity, ants[i].initialCity);
}
}
void updatePheromones() {
#pragma omp parallel for
for (int i = 0; i < N_POINTS; i++) {
for (int j = 0; j < N_POINTS; j++) {
float sumDist = 0;
for (int k = 0; k < N_AGENTS; k++) {
for (int q = 0; q < N_POINTS; q++) {
if (ants[k].route[q] == j) {
if (ants[k].route[q - 1] == i) {
sumDist += ants[k].pathLength;
}
break;
}
}
}
if (sumDist != 0)
pheromones[i][j] = (1 - P) * pheromones[i][j] + 1.0 / sumDist;
else
pheromones[i][j] = (1 - P) * pheromones[i][j];
}
}
}
int main() {
#pragma omp threadprivate(seed)
float prevAvg = 1e9;
float sum = 0;
int iter = 1; 
initVec();
initPheromones();
printf("INITIALIZED EVERYTHING\n");
do {
resetAgents();
releaseAgents();
updatePheromones();
minPathLength = ants[0].pathLength;
sum = ants[0].pathLength;
for (int i = 1; i < N_AGENTS; i++) {
if (ants[i].pathLength < minPathLength) {
minPathLength = ants[i].pathLength;
}
sum += ants[i].pathLength;
}
prevAvg = avgPathLength;
avgPathLength = sum / N_AGENTS;
iter++;
} while (abs(avgPathLength - prevAvg) / prevAvg > 0.01);
printf("Iterations: %d\tMin Path Length: %.2f\tAverage Path: %.2f\n", iter, minPathLength, avgPathLength);
return 0;
}