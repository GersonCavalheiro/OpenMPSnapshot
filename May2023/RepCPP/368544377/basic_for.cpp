#include "omp.h"

#include <iostream>
#include <string>



int main()
{
#pragma omp parallel for 
for(int i = 0 ; i < 32 ; ++i) 
{ 
std::cout << i << std::endl ; 
}
std::cout << "Back to serial" << std::endl ; 

#pragma omp parallel for collapse(2)
for(int i = 0 ; i < 2 ; ++i)
{
for(int j = 0 ; j < 2 ; ++j)
{
std::cout << omp_get_thread_num() ;
}
}
std::cout << ". I should print 4 unique threads, which would've all been created at the start to avoid weird overheads of two parallel regions"<< std::endl ;
return 0 ;
}
