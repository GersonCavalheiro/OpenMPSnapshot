#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
int search(int queenPlaced[], int N, int currentRow);
bool tryNewQueen();
int main(int argc, char *argv[]){
int N = atoi(argv[1]);
int nbThread = atoi(argv[2]);
int queenPlaced[N];
for (int i = 0; i < N ; i++){
queenPlaced[i] = -1;
}
int result = 0;
double start, end;
start = omp_get_wtime();
#pragma omp parallel num_threads (nbThread) private (queenPlaced)
{
#pragma omp for schedule (dynamic)
for (int i = 0; i < N; i++) {
queenPlaced[0] = i;
result += search(queenPlaced, N, 1);
}
}
end = omp_get_wtime();
printf("time : %f ", end - start);
printf("result = %d\n", result);
}
int search(int queenPlaced[], int N, int currentRow){
if(currentRow == N){
return 1;
}
int subResult = 0;
int newQueenPlaced[(currentRow+1)];
for (int i = 0; i < currentRow ; i++){
newQueenPlaced[i] = queenPlaced[i];
}
for (int i = 0; i < N ; i++){
if(tryNewQueen(queenPlaced, currentRow, i)){
newQueenPlaced[currentRow] = i;
subResult += search(newQueenPlaced, N, currentRow+1);
}
}
return subResult;
}
bool tryNewQueen(int queenPlaced[], int currentRow, int i){
for (int j = 0; j < currentRow; j++){
if (queenPlaced[j] == i || fabs(j - currentRow) == fabs(queenPlaced[j] - i)){
return false;
}
}
return true;
}