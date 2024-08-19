#include <omp.h>
#include <ctype.h>
#include <stdio.h>
#include <iostream>
using namespace std;
void find_max(int n, double* vector, int numThreads)
{
double max_element = vector[0];     
double itime, ftime, exec_time;     
omp_set_dynamic(0);                 
omp_set_num_threads(numThreads);    
itime = omp_get_wtime();            
#pragma omp parallel 
{
#pragma for reduction(max : max_element) 
for (int idx = 0; idx < n; idx++)
max_element = max_element > vector[idx] ? max_element : vector[idx];
}
ftime = omp_get_wtime();            
exec_time = ftime - itime;          
printf("Time taken %f\t", exec_time);   
printf(" Max value %f ", max_element);  
}
int main(int argc, char* argv[]) {
int n = atoi(argv[1]);
double* vector = new double[n];
for (long int i = 0; i <= n; i++) {
vector[i] = (double)rand() / RAND_MAX;
}  
for (int j=1; j<= 10; j++)
{
printf("\nNum of threads %d \t", j); 
find_max(n,vector,j);
}
delete[] vector;  
}