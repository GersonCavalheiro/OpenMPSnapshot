#include "kmeans.h"


int main(void)
{
std::chrono::time_point<std::chrono::system_clock> loop_benchmark, prev_loop_benchmark, start;
std::cout.precision(std::numeric_limits<float>::max_digits10);                                
prev_loop_benchmark = start = init_benchmark();                                               
std::vector<std::array<float, Nv>> Vec;                                                       
std::array<std::vector<int>, Nc> Classes;                                                     
std::vector<std::array<float, Nv>> old_Center;                                                
std::vector<std::array<float, Nv>> new_Center(Nc);                                            
long double iter_conv = std::numeric_limits<double>::infinity();                              
long double norm_iter_conv = std::numeric_limits<double>::infinity();                         
#pragma omp parallel num_threads(2)
{
#pragma omp master
{
Vec.reserve(N);                                                               
vec_init(Vec);                                                                
}
#pragma omp single
{
old_Center.reserve(Nc);                                                       
init_centers(old_Center);                                                     
}
}
std::chrono::time_point<std::chrono::system_clock> optimization_benchmark = init_benchmark(); 
for (int iter_counter = 1; norm_iter_conv > THR_KMEANS; iter_counter += 1)                    
{
std::chrono::duration<double> bench_res =
bench_loop(prev_loop_benchmark, loop_benchmark = init_benchmark());           
kmeans_progress(iter_counter, iter_conv, norm_iter_conv, bench_res, VERBOSITY);       
compute_classes(Vec, old_Center, Classes);                                            
optimize_center(Vec, new_Center, Classes);                                            
long double prev_iter_conv = iter_conv;                                               
iter_conv = convergence(new_Center, old_Center);                                      
norm_iter_conv = normalize_convergence(iter_conv, prev_iter_conv, iter_counter);      
old_Center.swap(new_Center);                                                          
}
std::chrono::time_point<std::chrono::system_clock> end = terminate_bench();                   
std::pair<std::chrono::duration<double>, std::chrono::duration<double>> bench_res =
benchmark_results(start, optimization_benchmark, end);                                
kmeans_termination(bench_res, norm_iter_conv);                                                
return 0;
}
