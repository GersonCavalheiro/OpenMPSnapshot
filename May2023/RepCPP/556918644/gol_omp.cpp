
#include <stdlib.h>
#include <stdio.h>
#include <chrono>
#include <omp.h>

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

#define THREADS 12

char analyzeCell(char *c_m, int N, int i, int j)
{
int alive_neighbours = 0, dead_neighbours = 0;

c_m[(i - 1) * N + (j - 1)] == '.' ? dead_neighbours++ : alive_neighbours++;
c_m[(i - 1) * N + j] == '.' ? dead_neighbours++ : alive_neighbours++;
c_m[(i - 1) * N + (j + 1)] == '.' ? dead_neighbours++ : alive_neighbours++;
c_m[i * N + (j + 1)] == '.' ? dead_neighbours++ : alive_neighbours++;
c_m[(i + 1) * N + (j + 1)] == '.' ? dead_neighbours++ : alive_neighbours++;
c_m[(i + 1) * N + j] == '.' ? dead_neighbours++ : alive_neighbours++;
c_m[(i + 1) * N + (j - 1)] == '.' ? dead_neighbours++ : alive_neighbours++;
c_m[i * N + (j - 1)] == '.' ? dead_neighbours++ : alive_neighbours++;

if (alive_neighbours < 2)
return '.';

else if (alive_neighbours > 3)
return '.';

else if (c_m[i * N + j] == 'X' && (alive_neighbours == 2 || alive_neighbours == 3))
return 'X';

else if (c_m[i * N + j] == '.' && alive_neighbours == 3)
return 'X';

else
return 'X';
}

void initMatrix(char *m, int N)
{
for (int i = 0; i < N; i++)
for (int j = 0; j < N; j++)
m[i * N + j] = rand() % 2 > 0 ? '.' : 'X';
}

void printMatrix(char *m, int N)
{
printf("\n");

for (int i = 1; i < N - 1; i++)
{
for (int j = 1; j < N - 1; j++)
printf("%c ", m[i * N + j]);
printf("\n");
}
}

int main(int argc, char **argv)
{
int N = 1 << atoi(argv[1]);

size_t bytes = N * N * sizeof(char);

char *c_m, *n_m;

c_m = (char *)malloc(bytes);
n_m = (char *)malloc(bytes);

initMatrix(c_m, N);


auto start = high_resolution_clock::now();

#pragma omp parallel for num_threads(THREADS)
for (int i = 1; i < N - 1; i++)
for (int j = 1; j < N - 1; j++)
n_m[i * N + j] = analyzeCell(c_m, N, i, j);

auto stop = high_resolution_clock::now();


duration<double, std::milli> ms_double = stop - start;

printf("\nCompleted successfully!\n");
printf("analyzeCell() execution time on the CPU: %f ms\n", ms_double.count());

free(c_m);
free(n_m);

return 0;
}