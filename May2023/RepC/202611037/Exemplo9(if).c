#include <stdio.h>  
#include <omp.h>  
void test(int val)  {  
#pragma omp parallel if (val)  
if (omp_in_parallel())   {  
#pragma omp single  
printf("Valor = %d, paralelizada com %d threads\n", val, omp_get_num_threads());  
}  
else  {  
printf("Valor = %d, serializada\n", val);  
}  
}  
int main( )   {  
omp_set_num_threads(2);  
test(0);  
test(2);  
} 