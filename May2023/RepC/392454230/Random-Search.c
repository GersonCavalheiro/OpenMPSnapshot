#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define N_POINTS 10000 
#define ITERATIONS 1e9 
float cities[N_POINTS][2] = {0};
int route[N_POINTS + 1] = {0};
float totDist = 0;
void initVec() {
for (int i = 0; i < N_POINTS; i++) {
route[i] = i;
cities[i][0] = (float)rand() / RAND_MAX * 1e3;
cities[i][1] = (float)rand() / RAND_MAX * 1e3;
}
route[N_POINTS] = 0;
}
float dist(int p1, int p2) {
float register dx = cities[p1][0] - cities[p2][0];
float register dy = cities[p1][1] - cities[p2][1];
return (float)sqrt(dx * dx + dy * dy);
}
void moveCity() {
int register index1, index2;
float tempDist = totDist;
do {
index1 = 1 + rand() % (N_POINTS - 1);
index2 = 1 + rand() % (N_POINTS - 1);
} while (index1 == index2);
int register point1 = route[index1];
int register point2 = route[index2];
if (abs(index1 - index2) == 1) 
{
if (index1 > index2) {
int tmp = index1;
index1 = index2;
index2 = tmp;
}
tempDist -= dist(route[index1], route[index1 - 1]);
tempDist -= dist(route[index2], route[index2 + 1]);
tempDist += dist(route[index1], route[index2 + 1]);
tempDist += dist(route[index2], route[index1 - 1]);
}
else 
{
tempDist -= dist(route[index1], route[index1 - 1]);
tempDist -= dist(route[index1], route[index1 + 1]);
tempDist -= dist(route[index2], route[index2 - 1]);
tempDist -= dist(route[index2], route[index2 + 1]);
tempDist += dist(route[index1], route[index2 - 1]);
tempDist += dist(route[index1], route[index2 + 1]);
tempDist += dist(route[index2], route[index1 - 1]);
tempDist += dist(route[index2], route[index1 + 1]);
}
if (tempDist < totDist) {
route[index1] = point2;
route[index2] = point1;
totDist = tempDist;
}
}
int main() {
initVec();
#pragma omp parallel for reduction(+:totDist)
for (int i = 0; i < N_POINTS; i++) {
totDist += dist(route[i], route[i + 1]);
}
printf("Starting total distance: %.2f\n", totDist);
float startDist = totDist;
for (int i = 0; i < ITERATIONS; i++) {
moveCity();
}
printf("Final total distance: %.2f\n", totDist);
printf("Delta: %.2f\n\n", totDist - startDist);
return 0;
}
