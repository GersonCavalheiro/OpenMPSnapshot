#include <iostream>
#include <cstring>
#include <omp.h>

constexpr int WIDTH = 20;
constexpr int HEIGHT = 30;
constexpr int NUM = 1000;

void print_matrix(int(*matrix)[WIDTH]) {
for (int i = 0; i < HEIGHT; i++) {
for (int j = 0; j < WIDTH; j++) {
printf("%4d ", matrix[i][j]);
}
printf("\n");
}
printf("\n");
}

void calc1(int(*matrix)[WIDTH]) {
#pragma omp parallel
{
#pragma omp for
for (int i = 0; i < HEIGHT; i++) {
for (int j = 0; j < WIDTH; j++) {
matrix[i][j] = omp_get_thread_num() * NUM + i * WIDTH + j;
}
}
}
}

void calc2(int(*matrix)[WIDTH]) {
#pragma omp parallel
{
#pragma omp for schedule(static, 1)
for (int i = 0; i < HEIGHT; i++) {
for (int j = 0; j < WIDTH; j++) {
matrix[i][j] = omp_get_thread_num() * NUM + i * WIDTH + j;
}
}
}
}

void calc3(int(*matrix)[WIDTH]) {
#pragma omp parallel
{
for (int i = 0; i < HEIGHT; i++) {
#pragma omp for schedule(static, 1)
for (int j = 0; j < WIDTH; j++) {
matrix[i][j] = omp_get_thread_num() * NUM + i * WIDTH + j;
}
}
}
}

int main() {
int dotMatrix[HEIGHT][WIDTH];

memset(dotMatrix, NULL, sizeof(dotMatrix));

omp_set_num_threads(3);

printf("1. parallel for \n\n");
calc1(dotMatrix);
print_matrix(dotMatrix);

printf("2.  ྿ round-robin  Ҵ\n\n");
calc2(dotMatrix);
print_matrix(dotMatrix);

printf("3.   round-robin  Ҵ\n\n");
calc3(dotMatrix);
print_matrix(dotMatrix);

return 0;
}