#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define N 100000           
#define Nv 1000            
#define Nc 100             
#define THRESHOLD 0.000001 
#define NUM_CORES 8
float vectors[N][Nv];
float centres[Nc][Nv];
int classes[N];
void initialiseVecs() {
for (int i = 0; i < N; i++) {
for (int j = 0; j < Nv; j++) {
vectors[i][j] = 1.0 * rand() / RAND_MAX;
}
}
}
void cpyVec(float *A, float *B) {
#pragma omp simd
for (int i = 0; i < Nv; i++) {
B[i] = A[i];
}
}
void initCentres() {
int temp[Nc];
int sel, flag = 0;
for (int i = 0; i < Nc; i++) {
sel = rand() % N;
for (int j = 0; j < i; j++) {
if (sel == temp[j]) {
flag = 1;
break;
}
}
if (flag == 1) {
flag = 0;
i--;
}
else {
temp[i] = sel;
cpyVec(&vectors[sel][0], &centres[i][0]);
}
}
}
float dist(float *A, float *B) {
float sum = 0;
for (int i = 0; i < Nv; i++) {
float x = A[i] - B[i];
sum += x * x;
}
return sum;
}
float computeClasses() {
float tempdist = 0;
float sumdists = 0;
#pragma omp parallel for private(tempdist) reduction(+:sumdists) shared(classes)
for (int i = 0; i < N; i++) {
float min = 1.0 * RAND_MAX;
for (int j = 0; j < Nc; j++) {
tempdist = dist(&vectors[i][0], &centres[j][0]);
if (tempdist < min) {
classes[i] = j;
min = tempdist;
}
}
sumdists += min;
}
return sumdists;
}
void addvec(float *A, float *B) {
#pragma omp simd
for (int i = 0; i < Nv; i++) {
B[i] += A[i];
}
}
void resetVec(float *A) {
#pragma omp simd
for (int i = 0; i < Nv; i++) {
A[i] = 0;
}
}
void computeCentres() {
float count[Nc] = {0};
for (int i = 0; i < Nc; i++) {
resetVec(&centres[i][0]);
}
for (int i = 0; i < N; i++) {
int tmp = classes[i];
count[tmp]++;
addvec(&vectors[i][0], &centres[tmp][0]);
}
#pragma omp parallel for shared(count) schedule(dynamic)
for (int i = 0; i < Nc; i++) {
if (count[i] == 0) {
printf("ERROR, category %d has no vectors.\n", i);
}
else {
float inv = 1 / count[i];
for (int j = 0; j < Nv; j++) {
centres[i][j] *= inv;
}
}
}
}
int main() {
float sumdist = 1e30, sumdistold;
int i = 0;
initialiseVecs();
initCentres();
do {
i++;
sumdistold = sumdist;
sumdist = computeClasses();
printf("Total distance in loop %d is %0.2f\n", i, sumdist);
computeCentres();
} while ((sumdistold - sumdist) / sumdistold > THRESHOLD);
return 0;
}
