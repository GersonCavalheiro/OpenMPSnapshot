#include <omp.h>
#include <iostream>

int main() {
int threadNumber;
#pragma omp parallel 
{
#pragma omp for collapse(2)
for (int i=0; i<4; i++) {
for (int j=0; j<4; j++) {
threadNumber = omp_get_thread_num();
std::cout << threadNumber;
}
}

}
}
