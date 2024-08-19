#include <stdio.h>
#include <stdlib.h>
#define MAXN 2005
#define countneighbors(cellmap, i, j) \
cellmap[i-1][j-1] + cellmap[i-1][j] + cellmap[i-1][j+1] + cellmap[i][j+1] + \
cellmap[i+1][j+1] + cellmap[i+1][j] + cellmap[i+1][j-1] + cellmap[i][j-1]
typedef unsigned long long ULL;
void print_map(char *cell[], char *cell_another[], int N){
for(int i = 1; i <= N; i++){
for(int j = 1; j <= N; j++){
printf("%d", cell[i][j]);
cell[i][j] = 0; 
cell_another[i][j] = 0;
}
printf("\n");
}
return;
}
void simulate(char **cell, char **cell_buffer, int N, int M){
char **cell_tmp = cell;
char **cell_buffer_tmp = cell_buffer;
char **tmp;
#pragma omp parallel firstprivate(cell_tmp, cell_buffer_tmp) private(tmp)
for(int m = 1; m <= M; m++){
#pragma omp for 
for(int i = 1; i <= N; i++){
for(int j = 1; j <= N; j++){
int neighbors = countneighbors(cell_tmp, i, j);
if(cell_tmp[i][j]){
cell_buffer_tmp[i][j] = neighbors == 2 || neighbors == 3;
}else{
cell_buffer_tmp[i][j] = (neighbors == 3);
}
}
}
tmp = cell_tmp;
cell_tmp = cell_buffer_tmp;
cell_buffer_tmp = tmp;
}
}
int main(){
char buf[MAXN];
char *cell[MAXN];
char *cell_buffer[MAXN];
int N, M;
for(int i = 0; i < MAXN; i++){
cell[i] = (char *)calloc(MAXN, sizeof(char));
cell_buffer[i] = (char *)calloc(MAXN, sizeof(char));
}
while(1){
fgets(buf, MAXN, stdin);
if(feof(stdin)) break;
sscanf(buf, "%d %d", &N, &M);
for(int i = 1; i <= N; i++){
fgets(buf, MAXN, stdin);
for(int j = 1; j <= N; j++){
cell[i][j] = buf[j-1]-'0';
}
}
simulate(cell, cell_buffer, N, M);
if((M & 1) == 1) print_map(cell_buffer, cell, N);
else print_map(cell, cell_buffer, N);
}
return 0;
}
