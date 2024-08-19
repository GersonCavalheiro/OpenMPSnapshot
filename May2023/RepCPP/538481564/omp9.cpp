#include <limits>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

using namespace std;

int main() {

const int WIDTH = 6;
const int HEIGHT = 8;

int min = numeric_limits<int>::max();
int max = numeric_limits<int>::min();

int d[WIDTH][HEIGHT];
srand(time(NULL));

for (int i = 0; i < WIDTH; i++) {
for (int j = 0; j < HEIGHT; j++) {
d[i][j] = rand();
printf("%d ", d[i][j]);
}
printf("\n");
}
printf("\n");

#pragma omp parallel num_threads(8)
for (int i = 0; i < WIDTH; i++) {

#pragma omp parallel for
for (int j = 0; j < HEIGHT; j++) {
if (d[i][j] > max) {

#pragma omp critical
max = d[i][j];
}
else if (d[i][j] < min) {

#pragma omp critical
min = d[i][j];
}
}
}
printf("Min value: %d\n", min);
printf("Max value: %d\n", max);
}
