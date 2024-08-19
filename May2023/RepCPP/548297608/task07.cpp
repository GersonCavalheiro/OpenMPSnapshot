
#include <iostream>
#include <ctime>
#include <omp.h>

int main() {
int TWELVE = 12;
std::srand(std::time(nullptr));

int a[TWELVE], b[TWELVE], c[TWELVE];

if (omp_get_dynamic() != 0) {
omp_set_dynamic(0);
}

#pragma omp parallel for schedule(static, 2) num_threads(3)
for (int i = 0; i < TWELVE; i++) {
a[i] = std::rand();
b[i] = std::rand();
std::cout << "Thread " + std::to_string(omp_get_thread_num())
+ " of " + std::to_string(omp_get_num_threads()) + ". "
+ "Result: a[" + std::to_string(i) + "] = " + std::to_string(a[i]) + ", "
+ "b[" + std::to_string(i) + "] = " + std::to_string(b[i]) + ". \n";
}

std::cout << "\n";

#pragma omp parallel for schedule(dynamic, 3) num_threads(4)
for (int i = 0; i < TWELVE; i++) {
c[i] = a[i] + b[i];
std::cout << "Thread " + std::to_string(omp_get_thread_num())
+ " of " + std::to_string(omp_get_num_threads()) + ". "
+ "Result: c[" + std::to_string(i) + "] = " + std::to_string(c[i]) + ". \n";
}

return 0;
}
