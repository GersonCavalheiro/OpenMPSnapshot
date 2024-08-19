#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
unsigned int solutions;
void setQueen(int queens[], int row, int col, int size) 
{
for(int i = 0; i < row; i++) {
if (queens[i] == col) {
return;
}
if (abs(queens[i] - col) == (row - i)) {
return;
}
}
queens[row] = col;
if(row == size - 1) {
#pragma omp atomic
solutions++;  
}
else {
for(int i = 0; i < size; i++) {
setQueen(queens, row + 1, i, size);
}
}
}
void solve(int size) 
{
for(int i = 0; i < size; i++) {
int *queens = malloc(sizeof(int)*size); 
setQueen(queens, 0, i, size);
free(queens);
}
}
int main(int argc, char* argv[])
{
double start_time, end_time;
if (argc != 2){
printf("ERROR! Usage: ./executable size\n");
return EXIT_FAILURE;
}
int size = atoi(argv[1]);
start_time = omp_get_wtime();
solve(size);
end_time = omp_get_wtime();
printf("Sequential Solution with a size of n = %d\n", size);
printf("The execution time is %g sec\n", end_time - start_time);
printf("Number of found solutions is %d\n", solutions);
return EXIT_SUCCESS;
}