
#include <iostream>
#include <omp.h>

int main() {
if (omp_get_dynamic() != 0) {
omp_set_dynamic(0);
}

omp_set_num_threads(3);
#pragma omp parallel if(omp_get_max_threads() > 1)
{
std::string s = "Hello, world from thread " + std::to_string(omp_get_thread_num())
+ " of " + std::to_string(omp_get_num_threads()) + ". \n";
std::cout << s;
}

omp_set_num_threads(1);
#pragma omp parallel if(omp_get_max_threads() > 1)
{
std::string s = "Hello, world from thread " + std::to_string(omp_get_thread_num())
+ " of " + std::to_string(omp_get_num_threads()) + ". \n";
std::cout << s;
}
return 0;
}
