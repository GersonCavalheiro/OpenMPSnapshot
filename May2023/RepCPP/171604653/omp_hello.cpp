#include <iostream>
#include <stdlib.h>
#include <omp.h> 

using namespace std;

int main(int argc, char* argv[])
{
double start_time = omp_get_wtime(); 

if(argc < 2)
{
cerr << "Please enter a command of the form: ./<prog_name> <num_of_threads>" << endl; 
return -1; 
}

int thread_count = atoi(argv[1]); 

#pragma omp parallel num_threads(thread_count) 
{
int thread_ID, total_threads; 
thread_ID = omp_get_thread_num(); 
total_threads = omp_get_num_threads(); 

#pragma omp critical 
{
if (thread_ID == 0) 
{
cout << "Threads are 0-indexed with the master thread as thread 0." << endl;
cout << "Total number of threads: " << total_threads << endl;
}

cout << "Hello from " << thread_ID << " of " << total_threads << "." << endl;
}
}

double run_time = omp_get_wtime() - start_time; 
cout << "Execution time: " << run_time << "seconds." << endl;
return 0;
}