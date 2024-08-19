#include <bits/stdc++.h>
#include <stdlib.h>
#include <omp.h>
#include <stdio.h>
#include <unistd.h>
#include <chrono> 
using namespace std;    
#define THREADS 4
#define N 16
int main ( ) {	
std::chrono::time_point<std::chrono::system_clock> start, end; 
start = std::chrono::system_clock::now();
int i;
#pragma omp parallel for schedule(dynamic) num_threads(THREADS)
for (i = 0; i < N; i++) {
sleep(i);
printf("Thread %d has completed iteration %d.\n", omp_get_thread_num( ), i);
}
printf("All done!\n");
end = std::chrono::system_clock::now(); 

std::chrono::duration<double> elapsed_seconds = end - start; 
std::time_t end_time = std::chrono::system_clock::to_time_t(end); 

std::cout<< "Time dynamic Loop: " << elapsed_seconds.count() <<endl;
return 0;
}

