#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#define THREADS 4
void doPuzzle(char **, int nRows, int nCols);
int main()
{
int nRows, nCols;
printf("Enter number of rows: ");
scanf("%d", &nRows);
printf("Enter number of columns: ");
scanf("%d", &nCols);
int i, j;
char **matrix = (char **)malloc(nRows * sizeof(char *));
for(i=0; i<nRows; ++i)
{
matrix[i] = (char *) malloc(nCols * sizeof(char *));
}
omp_set_num_threads(THREADS);
char letter;
for (i =0; i<nRows; ++i)
{
for (j=0; j<nCols; ++j)
{
printf("Row %d Column %d: ", i+1, j+1);
scanf(" %c", &letter);
while (letter != 'O' && letter != 'X')
{
printf("Invalid input.\n"
"Allowed characters are: 'X' or 'O'\n"
"Try again\n");
printf("Row %d Column %d: ", i+1, j+1);
scanf(" %c", &letter);
}
matrix[i][j] = letter;
}
printf("\n");
}
printf("============= Initial Input ===========\n");
for (i =0; i<nRows; ++i)
{
for (j=0; j<nCols; ++j)
{
printf("%c ", matrix[i][j]);
}
printf("\n");
}
doPuzzle(matrix, nRows, nCols);
printf("\n============= Final Output =============\n");
for (i =0; i<nRows; ++i)
{
for (j=0; j<nCols; ++j)
{
printf("%c ", matrix[i][j]);
}
printf("\n");
}
for (i=0; i<nRows; ++i)
free(matrix[i]);
free(matrix);
return 0;
}
void doPuzzle(char **matrix, int nRows, int nCols) {
char letter = 'O';
int i,j;
#pragma omp parallel shared(matrix) private(i,j)
{
for(i=0; i<nRows; ++i)
{
for (j=0; j<nCols-1; ++j)
{
if (matrix[i][j+1] == letter && j < nCols - 2)
{
if (matrix[i][j] == 'X' && matrix[i][j+2] == 'X')
{
matrix[i][j+1] = 'X';
printf("\n==>Change occurred on Row %d Col %d\n", i+1, j+2);
} else continue;
} else continue;
}
}
}
}