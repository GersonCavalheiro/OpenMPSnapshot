#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
char **allocateArray(int n, int init_flag)
{
int  i, j;
char *p, **array;
p = malloc(n*n*sizeof(char));		
array = malloc(n*sizeof(char*));
for (i = 0; i < n; i++){			
array[i] = &(p[i*n]);			
for (j = 0; j < n; j++){
if (init_flag == 0) array[i][j] = 0;			
else array[i][j] = rand() % 2;
}
}
return array;
}
void deleteArray(char ***array)
{
free(&((*array)[0][0]));			
free(*array);						
}
void show(char **array, int N)
{
int i, j;
printf("\n
for (i = 0; i < N; i++){
for (j = 0; j < N; j++){
if (array[i][j] == 0) printf("-");
else if (array[i][j] == 1) printf("X");
else printf("?");
}
printf("\n");
}
}
void evolve_sides(char **old_gen, char **new_gen, char **border, int N, int thread_count, int *allzeros, int *change)
{
int i, j, neighbors;
#pragma omp parallel num_threads(thread_count) default(none) shared(old_gen, new_gen, border, N, allzeros, change) private(j, neighbors)
for (j = 0; j < N; j++)
{															
neighbors = 0;
if ((j == 0) && (border[4][0] == 1)) neighbors++;				
else if ((j != 0) && (border[5][j-1] == 1)) neighbors++;
if (border[5][j] == 1) neighbors++;								
if ((j == N-1) && (border[6][0] == 1)) neighbors++;				
else if ((j != N-1) && (border[5][j+1] == 1)) neighbors++;
if ((j == N-1) && (border[7][0] == 1)) neighbors++;				
else if ((j != N-1) && (old_gen[0][j+1] == 1)) neighbors++;
if ((j == N-1) && (border[7][1] == 1)) neighbors++;				
else if ((j != N-1) && (old_gen[1][j+1] == 1)) neighbors++;
if (old_gen[1][j] == 1) neighbors++;							
if ((j == 0) && (border[3][1] == 1)) neighbors++;				
else if ((j != 0) && (old_gen[1][j-1] == 1)) neighbors++;
if ((j == 0) && (border[3][0] == 1)) neighbors++;				
else if ((j != 0) && (old_gen[0][j-1] == 1)) neighbors++;
if ((old_gen[0][j] == 1) && ((neighbors < 2) || (neighbors > 3)))
new_gen[0][j] = 0;
else if ((old_gen[0][j] == 0) && (neighbors == 3))
new_gen[0][j] = 1;
else
new_gen[0][j] = old_gen[0][j];		
if (new_gen[0][j] != 0) *allzeros = 1;
if (new_gen[0][j] != old_gen[0][j]) *change = 1;
neighbors = 0;
if ((j == 0) && (border[3][N-2] == 1)) neighbors++;				
else if ((j != 0) && (old_gen[N-2][j-1] == 1)) neighbors++;
if (old_gen[N-2][j] == 1) neighbors++;							
if ((j == N-1) && (border[7][N-2] == 1)) neighbors++;			
else if ((j != N-1) && (old_gen[N-2][j+1] == 1)) neighbors++;
if ((j == N-1) && (border[7][0] == 1)) neighbors++;				
else if ((j != N-1) && (old_gen[N-1][j+1] == 1)) neighbors++;
if ((j == N-1) && (border[0][0] == 1)) neighbors++;				
else if ((j != N-1) && (border[1][j+1] == 1)) neighbors++;
if (border[1][j] == 1) neighbors++;								
if ((j == 0) && (border[2][0] == 1)) neighbors++;				
else if ((j != 0) && (border[1][j-1] == 1)) neighbors++;
if ((j == 0) && (border[3][0] == 1)) neighbors++;				
else if ((j != 0) && (old_gen[N-1][j-1] == 1)) neighbors++;
if ((old_gen[N-1][j] == 1) && ((neighbors < 2) || (neighbors > 3)))
new_gen[N-1][j] = 0;
else if ((old_gen[N-1][j] == 0) && (neighbors == 3))
new_gen[N-1][j] = 1;
else
new_gen[N-1][j] = old_gen[N-1][j];		
if (new_gen[N-1][j] != 0) *allzeros = 1;
if (new_gen[N-1][j] != old_gen[N-1][j]) *change = 1;
}
#pragma omp parallel num_threads(thread_count) default(none) shared(old_gen, new_gen, border, N, allzeros, change) private(i, neighbors)
for (i = 0; i < N; i++)
{															
neighbors = 0;
if ((i == 0) && (border[4][0] == 1)) neighbors++;				
else if ((i != 0) && (border[3][i-1] == 1)) neighbors++;
if ((i == 0) && (border[5][0] == 1)) neighbors++;				
else if ((i != 0) && (old_gen[i-1][0] == 1)) neighbors++;
if ((i == 0) && (border[5][1] == 1)) neighbors++;				
else if ((i != 0) && (old_gen[i-1][1] == 1)) neighbors++;
if (old_gen[i][1] == 1) neighbors++;							
if ((i == N-1) && (border[1][1] == 1)) neighbors++;				
else if ((i != N-1) && (old_gen[i+1][1] == 1)) neighbors++;
if ((i == N-1) && (border[1][0] == 1)) neighbors++;				
else if ((i != N-1) && (old_gen[i+1][0] == 1)) neighbors++;
if ((i == N-1) && (border[2][0] == 1)) neighbors++;				
else if ((i != N-1) && (border[3][i+1] == 1)) neighbors++;
if (border[3][i] == 1) neighbors++;								
if ((old_gen[i][0] == 1) && ((neighbors < 2) || (neighbors > 3)))
new_gen[i][0] = 0;
else if ((old_gen[i][0] == 0) && (neighbors == 3))
new_gen[i][0] = 1;
else
new_gen[i][0] = old_gen[i][0];		
if (new_gen[i][0] != 0) *allzeros = 1;
if (new_gen[i][0] != old_gen[i][0]) *change = 1;
neighbors = 0;
if ((i == 0) && (border[5][N-2] == 1)) neighbors++;				
else if ((i != 0) && (old_gen[i-1][N-2] == 1)) neighbors++;
if ((i == 0) && (border[5][N-1] == 1)) neighbors++;				
else if ((i != 0) && (old_gen[i-1][N-1] == 1)) neighbors++;
if ((i == 0) && (border[6][0] == 1)) neighbors++;				
else if ((i != 0) && (border[7][i-1] == 1)) neighbors++;
if (border[7][i] == 1) neighbors++;								
if ((i == N-1) && (border[0][0] == 1)) neighbors++;				
else if ((i != N-1) && (border[7][i+1] == 1)) neighbors++;
if ((i == N-1) && (border[1][N-1] == 1)) neighbors++;			
else if ((i != N-1) && (old_gen[i+1][N-1] == 1)) neighbors++;
if ((i == N-1) && (border[1][N-2] == 1)) neighbors++;			
else if ((i != N-1) && (old_gen[i+1][N-2] == 1)) neighbors++;
if (old_gen[i][N-2] == 1) neighbors++;							
if ((old_gen[i][N-1] == 1) && ((neighbors < 2) || (neighbors > 3)))
new_gen[i][N-1] = 0;
else if ((old_gen[i][N-1] == 0) && (neighbors == 3))
new_gen[i][N-1] = 1;
else
new_gen[i][N-1] = old_gen[i][N-1];		
if (new_gen[i][N-1] != 0) *allzeros = 1;
if (new_gen[i][N-1] != old_gen[i][N-1]) *change = 1;
}
}
void evolve_inner(char **old_gen, char **new_gen, int N, int thread_count, int *allzeros, int *change)
{
int i, j, up, down, left, right, neighbors;
#pragma omp parallel num_threads(thread_count) default(none) shared(old_gen, new_gen, N, allzeros, change) private(i, j, up, down, left, right, neighbors)
for (i = 1; i < N-1; i++)
{										
up   = i-1;
down = i+1;
#pragma omp for
for (j = 1; j < N-1; j++)
{									
left  = j-1;
right = j+1;
neighbors = 0;
if (old_gen[up][left]    == 1) neighbors++;				
if (old_gen[up][j]       == 1) neighbors++;
if (old_gen[up][right]   == 1) neighbors++;
if (old_gen[i][right]    == 1) neighbors++;
if (old_gen[down][right] == 1) neighbors++;
if (old_gen[down][j]     == 1) neighbors++;
if (old_gen[down][left]  == 1) neighbors++;
if (old_gen[i][left]     == 1) neighbors++;
if ((old_gen[i][j] == 1) && ((neighbors < 2) || (neighbors > 3)))
new_gen[i][j] = 0;
else if ((old_gen[i][j] == 0) && (neighbors == 3))
new_gen[i][j] = 1;
else
new_gen[i][j] = old_gen[i][j];		
if (new_gen[i][j] != 0) *allzeros = 1;
if (new_gen[i][j] != old_gen[i][j]) *change = 1;
}
}
}