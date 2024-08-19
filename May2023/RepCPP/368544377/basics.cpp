#include "omp.h"

#include <iostream>
#include <string>



int main()
{
#pragma omp parallel 
{ 
std::cout << "Hi!" << std::endl ; 
}
std::cout << "Back to serial" << std::endl ; 
#pragma omp parallel num_threads(4)
{
std::cout << "I should print 4 times, I've been told to!" << std::endl ;
}

#pragma omp parallel num_threads(4)
{
#pragma omp master
{
std::cout << "I'm in a threadpool of 4 but im the master thread, I only run once!" << std::endl ;
}
}

#pragma omp parallel num_threads(4)
{
#pragma omp critical
{
std::cout << "I'm in a threadpool of 4 but im a critical section so everything stops so I can run!" << std::endl ;
}
}

return 0 ;
}
