#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h> 
#include <string.h>
#include <omp.h>
#define N 14
bool SOLUTION_EXISTS = false;
bool can_be_placed(int board[N][N], int row, int col);
void print_solution(int board[N][N]);
bool solve_NQueens(int board[N][N], int col);
bool can_be_placed(int board[N][N], int row, int col) 
{ 
int i, j; 
for (i = 0; i < col; i++) 
if (board[row][i]) 
return false; 
for (i=row, j=col; i>=0 && j>=0; i--, j--) 
if (board[i][j]) 
return false; 
for (i=row, j=col; j>=0 && i<N; i++, j--) 
if (board[i][j]) 
return false; 
return true; 
} 
void print_solution(int board[N][N]) 
{ 
static int k = 1; 
printf("Solution #%d-\n",k++); 
int i,j;
for (i = 0; i < N; i++) 
{ 
for (j = 0; j < N; j++) 
printf(" %d ", board[i][j]); 
printf("\n"); 
} 
printf("\n"); 
}
bool solve_NQueens(int board[N][N], int col) 
{ 
int new_board[N][N];
memcpy(new_board, board, sizeof(new_board));
if (col == N) 
{ 
#pragma omp critical
SOLUTION_EXISTS = true;
return true; 
} 
int i;
for (i = 0; i < N; i++) 
{ 
if (can_be_placed(board, i, col)) 
{ 
new_board[i][col] = 1; 
#pragma omp task firstprivate(col) if(col < 3)
solve_NQueens(new_board, col + 1); 
new_board[i][col] = 0;
}
}
return SOLUTION_EXISTS; 
} 
int main() 
{ 
int board[N][N]; 
memset(board, 0, sizeof(board)); 
double time1 = omp_get_wtime();
#pragma omp parallel
{   
#pragma omp single
{       
#pragma omp taskgroup
{
solve_NQueens(board, 0);
}
}
}
if (SOLUTION_EXISTS == false) 
{ 
printf("No Solution Exits! \n"); 
} 
printf("Elapsed time: %0.2lf\n", omp_get_wtime() - time1); 
return 0;
} 
