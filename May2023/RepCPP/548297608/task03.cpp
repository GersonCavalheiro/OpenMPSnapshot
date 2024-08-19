
#include <iostream>
#include <omp.h>

int main() {

int a = 0;
int b = 0;

std::cout << "a = " + std::to_string(a) + "; b = " + std::to_string(b) + ". \n\n";

#pragma omp parallel num_threads(2) firstprivate(b) private(a)
{
a = 0;
a += omp_get_thread_num();
b += omp_get_thread_num();

std::cout << "a = " + std::to_string(a) + "; b = " + std::to_string(b) + ". \n";
}

std::cout <<  "\na = " + std::to_string(a) + "; b = " + std::to_string(b) + ". \n\n";

#pragma omp parallel num_threads(4) shared(a) private(b)
{
b = 0;
#pragma omp atomic
a -= omp_get_thread_num();
b -= omp_get_thread_num();

std::cout << "a = " + std::to_string(a) + "; b = " + std::to_string(b) + ". \n";
}

std::cout << "\na = " + std::to_string(a) + "; b = " + std::to_string(b) + ". \n";
return 0;
}
