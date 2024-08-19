#include "omp.h"

#include <iostream>
#include <string>
#include <cstddef>



int main()
{
#pragma omp parallel for schedule(static, 2)
for(std::size_t i = 0 ; i < 32 ; ++i)
{
std::cout << "Hi, I should be ran 16 times on: " << omp_get_thread_num() << std::endl ;
}
return 0 ;
}
