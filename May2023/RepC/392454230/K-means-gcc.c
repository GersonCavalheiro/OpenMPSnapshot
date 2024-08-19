#include <stdio.h>
#include <stdlib.h>
#pragma GCC optimize("O3", "unroll-loops", "omit-frame-pointer", "inline", "unsafe-math-optimizations");
#pragma GCC option("arch=native", "tune=native", "no-zero-upper");
#define N 100000           
#define Nv 1000            
#define Nc 100             
#define THRESHOLD 0.000001 
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
float dist(float *restrict A, float *restrict B) {
float sum = 0;
for (int i = 0; i < Nv; i++) {
sum += (A[i] - B[i]) * (A[i] - B[i]);
}
return sum;
}
float computeClasses() {
float tempdist = 0;
float sumdists = 0;
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
for (int i = 0; i < Nv; i++) {
B[i] += A[i];
}
}
void resetVec(float *A) {
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
computeCentres();
} while ((sumdistold - sumdist) / sumdistold > THRESHOLD);
printf("Total distance in loop %d is %0.2f\n", i, sumdist);
return 0;
}
