#include <iostream>
#include <omp.h>
#include <cstdlib>
#include <vector>
#include <algorithm>

constexpr int SIZE = 20;

int main() {
std::vector<int> arr(SIZE);
for (int &x:arr) x = rand();
for (int& x : arr) std::cout << x << ' ';
std::cout << '\n';

double start, end;

omp_set_num_threads(4);
start = omp_get_wtime();

for (int i = 0; i < SIZE; i++) {
if (i % 2 == 0) {
#pragma omp parallel for
for (int j = 0; j < SIZE - 1; j += 2) {
if (arr[j] > arr[j + 1]) std::swap(arr[j + 1], arr[j]);
}
}
else {
#pragma omp parallel for
for (int j = 1; j < SIZE - 1; j += 2) {
if (arr[j] > arr[j + 1]) std::swap(arr[j + 1], arr[j]);
}
}

}

end = omp_get_wtime();
std::cout << "Execute time " << end - start << "seconds\n";

std::cout << '\n';
for (int i = 0; i < SIZE; i++) std::cout << arr[i] << ' ';
return 0;
}