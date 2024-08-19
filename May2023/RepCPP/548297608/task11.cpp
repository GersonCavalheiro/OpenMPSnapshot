
#include <iostream>
#include <ctime>

int main() {
std::srand(std::time(nullptr));

int L = 777;
int N = 4;
int D = 7;
int a[L];
for (int l = 0; l < L; l++) {
a[l] = std::rand();
}

int max = INT_MIN;
bool isChanged = false;

#pragma omp parallel for num_threads(N) shared(max, isChanged)
for (int l = 0; l < L; l++) {
#pragma omp critical
if (a[l] % D == 0 && max < a[l]) {
max = a[l];
isChanged = true;
}
}

if (!isChanged) {
std::cout << "There are no multiples of " + std::to_string(D) + " in the array \n";
} else {
std::cout << "Max value is " + std::to_string(max) + "\n";
}

return 0;
}
