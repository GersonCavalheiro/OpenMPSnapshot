#include <stdio.h>
#include <omp.h>
int main() {
printf("==========EXERCISE 5d==========\n");
printf("    CRITICAL CONSTRUCT\n\t\tS. DHANYA ABHIRAMI\n\t\t16BCE0965\n");
int data;
#pragma omp parallel num_threads(3)
{
int id = omp_get_thread_num();    
int total = omp_get_num_threads();
#pragma omp critical 
{ 
data = id; 
printf("Greetings from process %d out of %d with Data %d\n", id, total, data);
}
}
printf("parallel for ends.\n");
return 0;
}