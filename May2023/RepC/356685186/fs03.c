#pragma GCC optimize("O3","unroll-loops","omit-frame-pointer","inline") 
#pragma GCC option("arch=native","tune=native","no-zero-upper") 
#pragma GCC target("avx")  
#include "stdio.h"
#include "stdlib.h"
#include "omp.h"
#include "stdbool.h"
#define N 35
void Initialization(int posY[N]) {
for (int i=0; i<N; i++) {
posY[i] = 0;
}
}
void Map2(int posY[N], int M) {
printf("\n\n========================\n"); 
printf("---- SOLUTION FOUND ----\n");
printf("========================\n");
for (int i=0; i<N; i++) printf("---"); printf("---\n##|");
for (int i=0; i<N; i++) printf("%2d ", i+1); printf("\n---");
for (int i=0; i<N; i++) printf("---"); printf("\n");
for (int y=0; y<N; y++) {
printf("%2d| ", y+1);
for (int x=0; x<N; x++) {
bool flag = false;
for (int i=0; i<M; i++) {
if (i==x && posY[i]==y) {
flag = true;
}
}
if (flag) printf("Q");
else printf("~");
printf("  ");
}
printf("\n");
}
for (int i=0; i<N; i++) printf("---"); printf("---\n\n");
}
bool isSafeFromPrevious(int posY[N], int x, int y) {
int currentQueen = x;
for (int oldQueen=0; oldQueen<currentQueen; oldQueen++) {
if (oldQueen==x || posY[oldQueen]==y) return false; 
else if (y==posY[oldQueen]+(currentQueen-oldQueen) || y==posY[oldQueen]-(currentQueen-oldQueen)) return false; 
}
return true;
}
bool SolveColumn(int posY[N], int current, int startingPoint) {
int x = current;
for (int y=startingPoint; y<N; y++) {
if (isSafeFromPrevious(posY, x, y)) {
posY[current] = y;
return true;
}
}
return false;
}
int main() {
printf("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
printf("This program implements my backtracking method of solving the \"N-Queens Problem\".\n");
printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n");
printf("Now solving the problem. Please wait...\n");
int maxQueen = 0, threadSolved = -1;
long int finalSteps = 0, steps = 0;
int finalPosY[N] = {0};
#pragma omp parallel shared(finalSteps, finalPosY, threadSolved, maxQueen) default(none) reduction(+:steps)
{
int posY[N] = {0}; 
int yMin[N] = {0}; 
int i = 1;
bool canContinue = true;
posY[0] = (omp_get_thread_num()*N+1)/omp_get_num_threads();    
#pragma omp critical
printf("> I am thread %2d and my first queen will be at position: [ 0 , %2d]\n", omp_get_thread_num(), posY[0]);  
#pragma omp barrier
while(threadSolved<0 && canContinue) {
steps++;
if (!SolveColumn(posY, i, yMin[i])) {
yMin[i--] = 0;
yMin[i] = posY[i] + 1;
if (yMin[i] == N) i-=1;
if (i<0 || (i==0 && yMin[0] == N)) {
canContinue = false;
}
} else {
i++;
if (i==N-1) {
#pragma omp critical
{
threadSolved = omp_get_thread_num();
for (int w=0; w<N; w++) 
finalPosY[w] = posY[w];
finalSteps = steps;
maxQueen = i;
}
} else if (i>maxQueen) maxQueen = i;
}
#pragma omp master
printf("\r>> Current furthest queen reached is: %d", maxQueen);
}
}
printf("\n\nAlgorithm finished.\n");
printf("> Solution found by Thread #%d in %ld steps\n", threadSolved, finalSteps);
printf("> Sum of all threads's steps: %ld\n", steps);
Map2(finalPosY, N); 
bool allSafe = true;
for (int i=0; i<N-1; i++) {
if (!isSafeFromPrevious(finalPosY, i, finalPosY[i])) {
printf("WRONG SOLUTION!!\n");
allSafe = false;
}
}
if (allSafe) printf("The solution was validated and is accepted.\n");
}
