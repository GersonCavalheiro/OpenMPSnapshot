#include <omp.h>  
#include <stdio.h>  
int main() {
printf("==========EXERCISE 5c==========\n");
printf("    BARRIER CONSTRUCT\n\t\tS. DHANYA ABHIRAMI\n\t\t16BCE0965\n");
int n=10,i;
#pragma omp parallel private(i) shared(n) 
{
#pragma omp for
for (i=0; i<n; i++)
printf("%d\t",omp_get_thread_num() );
printf("\n");
#pragma omp for
for (i=0; i<n; i++)
printf("%d\t",omp_get_thread_num() );
}
printf("\n");
return (0);
}