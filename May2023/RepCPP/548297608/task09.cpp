
#include <iostream>
#include <ctime>

int main() {
std::srand(std::time(nullptr));

int I = 6;
int J = 8;
int d[I][J];
for (int i = 0; i < I; i++) {
for (int j = 0; j < J; j++) {
d[i][j] = std::rand();
}
}

int min = INT_MAX;
int max = INT_MIN;

#pragma omp parallel for num_threads(I)
for (int i = 0; i < I; i++) {
for (int j = 0; j < J; j++) {
if (min > d[i][j]) {
#pragma omp critical
min = d[i][j];
}
if (max < d[i][j]) {
#pragma omp critical
max = d[i][j];
}
}
}

std::cout << "Max value is " + std::to_string(max) + "\n"
+ "Min value is " + std::to_string(min) + "\n";
return 0;
}
