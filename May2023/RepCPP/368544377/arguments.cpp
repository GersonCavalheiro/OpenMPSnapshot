#include "omp.h"

#include <iostream>
#include <string>



int main()
{
int var = 13;

#pragma omp parallel num_threads(1)
{
std::cout << "implicit shared (DEFAULT): chances are this number is 13: " << var << std::endl ; 
}
#pragma omp parallel num_threads(1) shared(var)
{
std::cout << "explicit shared: chances are this number is 13: " << var << std::endl ; 
}

#pragma omp parallel private(var) num_threads(1)
{
std::cout << "private: chances are this number is not 13: " << var << std::endl ; 
}

#pragma omp parallel firstprivate(var) num_threads(1)
{
std::cout << "firstprivate: chances are this number is 13: " << var << std::endl ; 
}

#pragma omp parallel for lastprivate(var) num_threads(1)
for(int i = 0 ; i < 1 ; ++i)
{
std::cout << "lastprivate: chances are this number is 13: " << var << std::endl ; 
}
std::cout << "GLOBAL CONTEXT: Chances are lastprivate change the value from 13: " << var << std::endl ;

return 0 ;
}
