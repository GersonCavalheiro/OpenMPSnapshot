
#include <iostream>

int main() {

int a[10] = {5, 6, -2, 3, 4, 1, -1, -4, 8, 1};
int b[10] = {1, 7, 4, 0, -3, 8, 2, -1, 3, 3};

int sumA = 0, sumB = 0;

#pragma omp parallel for reduction(+: sumA) reduction(+: sumB)
for (int i = 0; i < 10; i++) {
sumA += a[i];
sumB += b[i];
}

std::cout << "A-array's average value is " + std::to_string(sumA / 10.0)
+ ". B-array's average value is " + std::to_string(sumB / 10.0) + ". \n";


if (sumA < sumB)
std::cout << "A-array's average value < B-array's average value";
else if (sumA > sumB)
std::cout << "A-array's average value > B-array's average value";
else
std::cout << "A-array's average value = B-array's average value";

return 0;
}
