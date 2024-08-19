
#include <iostream>
#include <omp.h>

int main() {
#pragma omp parallel num_threads(8)
{
std::string s = "Hello, world from thread " + std::to_string(omp_get_thread_num())
+ " of " + std::to_string(omp_get_num_threads()) + ". \n";
std::cout << s;
}
return 0;
}
