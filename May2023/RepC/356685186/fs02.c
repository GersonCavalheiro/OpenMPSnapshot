#include "stdio.h"
#include "stdlib.h"
#include "stdbool.h"
#define N 30
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
__time_t seconds;
srand(seconds);
int posY[N] = {0}; 
int yMin[N] = {0}; 
printf("Now solving the problem. Please wait...\n");
int i = 1, steps = 0;
while(i<N) {
steps++;
if (steps%10000==0) printf("\rStep: %3d, Queen: %2d", steps, i);
if (!SolveColumn(posY, i, yMin[i])) {
yMin[i--] = 0;
yMin[i] = posY[i] + 1;
if (yMin[i] == N) i-=1;
if (i<0 || (i==0 && yMin[0] == N)) {
printf("Couldn't find a solution\n");
exit(0);
}
} else {
i++;
}
}
printf("\rStep: %3d, Queen: %2d", steps, i);
Map2(posY, N); 
bool allSafe = true;
for (int i=0; i<N-1; i++) {
if (!isSafeFromPrevious(posY, i, posY[i])) {
printf("WRONG SOLUTION!!\n");
allSafe = false;
}
}
if (allSafe) printf("The solution was validated and is accepted.\n");
}
