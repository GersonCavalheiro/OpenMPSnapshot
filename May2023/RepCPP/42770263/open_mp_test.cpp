#include <iostream>
#include <omp.h>

int main() {
omp_set_num_threads(5);
#pragma omp parallel for
for (int i = 0; i < 100; i++) {
std::cout << "thread " << omp_get_thread_num() << ": " << i << std::endl;
}

}
