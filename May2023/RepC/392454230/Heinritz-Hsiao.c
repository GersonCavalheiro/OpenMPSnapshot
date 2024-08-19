#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define N_POINTS 10000
#define THRESHOLD 0.8
float cities[N_POINTS][2] = {0};  
short city_flags[N_POINTS] = {0}; 
float totDist = 0;                
int curr_index = 0;               
void initVec() {
for (int i = 0; i < N_POINTS; i++) {
city_flags[i] = 1;
cities[i][0] = (float)rand() / RAND_MAX * 1e3;
cities[i][1] = (float)rand() / RAND_MAX * 1e3;
}
}
float dist(int p1, int p2) {
float register dx = cities[p1][0] - cities[p2][0];
float register dy = cities[p1][1] - cities[p2][1];
return (float)sqrt(dx * dx + dy * dy);
}
float moveCity() {
int index1, index2;
float register mindist1 = 100e3;
float register mindist2 = 100e3;
#pragma omp parallel for
for (int i = 1; i < N_POINTS; i++) {
if (city_flags[i] == 1) {
float register tmpDist = dist(curr_index, i);
#pragma omp critical
{
if (tmpDist < mindist1) {
mindist2 = mindist1;
index2 = index1;
index1 = i;
mindist1 = tmpDist;
}
else if (tmpDist < mindist2) {
mindist2 = tmpDist;
index2 = i;
}
}
}
}
if ((float)rand() / RAND_MAX < THRESHOLD) {
city_flags[index1] = 0;
curr_index = index1;
return mindist1;
}
else {
city_flags[index2] = 0;
curr_index = index2;
return mindist2;
}
}
int main() {
initVec();
totDist += moveCity();
for (int i = 0; i < N_POINTS - 2; i++) {
totDist += moveCity();
}
totDist += dist(curr_index, 0);
printf("Final total distance: %.2f\n", totDist);
return 0;
}
