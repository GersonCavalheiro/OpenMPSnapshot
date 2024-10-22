#include <bits/stdc++.h>
#include <chrono> 
using namespace std;    
#define THREADS 10
#define N 100000000
int main ( ) {	
std::chrono::time_point<std::chrono::system_clock> start, end; 
start = std::chrono::system_clock::now();
int i;
printf("Running %d iterations on %d threads dynamically.\n", N, THREADS);
#pragma omp parallel for schedule(dynamic),num_threads(THREADS)
for (i = 0; i < N; i++) {
}
printf("All done!\n"); 
end = std::chrono::system_clock::now(); 

std::chrono::duration<double> elapsed_seconds = end - start; 
std::time_t end_time = std::chrono::system_clock::to_time_t(end); 

std::cout<< "Time dynamic: " << elapsed_seconds.count() <<endl;
return 0;
}
