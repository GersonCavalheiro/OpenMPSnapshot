#include "omp.h"

#include <iostream>



#pragma omp declare reduction (add_func : int : \
omp_out = omp_in + omp_out \
)

int main()
{

int reduction = 0 ;
#pragma omp parallel for reduction(add_func : reduction)
for(std::size_t i = 0 ; i < 10 ; ++i)
{
reduction = i ;
}
std::cout << "I should have a value of 0+1+2+3+4+5+6+7+8+9 (45)!: " << reduction << std::endl ;


int total = 0 ;
#pragma omp parallel for
for(std::size_t i = 0 ; i < 10 ; ++i)
{
#pragma omp atomic update
++total ;
}
std::cout << "I should have the value of 1+1+1+1+1+1+1+1+1+1+1 (10): " << total << std::endl ;

return 0 ;
}
