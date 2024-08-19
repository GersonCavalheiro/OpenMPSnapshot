#include <stdio.h>
#include <omp.h>
int main()
{
int i;
#pragma omp parallel num_threads(2) 
{
int tid = omp_get_thread_num(); 
#pragma omp for ordered schedule(static) 
for(i = 1; i <= 3; i++)
{
printf("[PRINT1] T%d = %d \n",tid,i);
printf("[PRINT2] T%d = %d \n",tid,i);
}
}
}
<<<<<<< HEAD
=======
>>>>>>> 3f1a2a26671abb5caaf3af4ac68667eb6a418f74
