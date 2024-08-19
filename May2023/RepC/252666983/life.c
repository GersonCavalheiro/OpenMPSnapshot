
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <omp.h>

#define SIZE 511
#define ITERATIONS 256
#define SCALE 2
#define FILENAME "life%3.3d.pgm"
#define CHUNKSIZE 1

#define MOD(a, b) (((a) + (b)) % (b))

typedef int (*generator_t)(int, int, int);

void writePgm(int **grid, int size, int scale, char *fileName) {

int x, y;
FILE *file = fopen(fileName, "w");

fprintf(file, "P2\n%4d %4d\n1\n", size * scale, size * scale);

for(x = 0; x < size * scale; x++) {
for(y = 0; y < size * scale; y++) {
fprintf(file, "%d\n", grid[x / scale][y / scale]);
}
}
fclose(file);
}

int** createGrid(int size, generator_t f) {

int x, y;
int **grid = malloc(size * sizeof(int*));

for(x = 0; x < size; x++) {
grid[x] = malloc(size * sizeof(int));

for(y = 0; y < size; y++) {
grid[x][y] = f(size, x, y);
}
}
return grid;
}

void freeGrid(int **grid, int size) {

int x;
for(x = 0; x < size; x++) {
free(grid[x]);
}
free(grid);
}

int adjacent(int **grid, int size, int x, int y) {

int xx, yy, adj = 0;

for(xx = -1; xx <= 1; xx++) {
for(yy = -1; yy <= 1; yy++) {

if(xx != 0 || yy != 0) {
adj += grid[MOD(x + xx, size)][MOD(y + yy, size)];
}
}
}
return adj;
}

void life(int **grid1, int **grid2, int size) {

int x, y, adj;

#pragma omp parallel for shared(grid1, grid2, size) private(x, y, adj) schedule(dynamic, CHUNKSIZE)
for(x = 0; x < size; x++) {
for(y = 0; y < size; y++) {

adj = adjacent(grid1, size, x, y);

if(adj < 2 || adj > 3) grid2[x][y] = 0;
if(adj == 2) grid2[x][y] = grid1[x][y];
if(adj == 3) grid2[x][y] = 1;
}
}
}

int EMPTY(int size, int x, int y) {
return 0;
}

int CROSS(int size, int x, int y) {
return x == size / 2 || y == size / 2;
}

long timeMs() {
struct timeval time;
gettimeofday(&time, NULL);
return time.tv_sec * 1000 + time.tv_usec / 1000;
}

int main(int argc, int **argv) {

int i; long t;
char fileName[20];
int **grid1 = createGrid(SIZE, CROSS),
**grid2 = createGrid(SIZE, EMPTY), **swap;

sprintf(fileName, FILENAME, 0);
writePgm(grid1, SIZE, SCALE, fileName);

t = timeMs();

for(i = 0; i < ITERATIONS; i++) {
life(grid1, grid2, SIZE);
swap = grid1; grid1 = grid2; grid2 = swap;
}

printf("Completed %d Game of Life iterations on a %dx%d grid in %dms.\n",
ITERATIONS, SIZE, SIZE, timeMs() - t);

sprintf(fileName, FILENAME, ITERATIONS);
writePgm(grid1, SIZE, SCALE, fileName);

freeGrid(grid1, SIZE); freeGrid(grid2, SIZE);
return 0;
}
