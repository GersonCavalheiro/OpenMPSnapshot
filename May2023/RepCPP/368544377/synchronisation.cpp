#include <iostream>

namespace omp {
#include "omp.h"
}



int main()
{

#pragma omp parallel
{
int mytid = omp::omp_get_thread_num() ;
#pragma omp barrier 
std::cout << "im thread#" << mytid << std::endl ;
}


#pragma omp parallel
{
#pragma omp critical 
std::cout << "Hi" << std::endl ; 
}
return 0 ;
}
