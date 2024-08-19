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
#pragma omp parallel num_threads(4)
{
#pragma omp master
{
multidimentional_float_vector_interface(Vec, "Vec");                                             
}
#pragma omp single
{
export_multidimentional_float_vector(Vec, "Vec");                                                
}
#pragma omp single
{
multidimentional_float_vector_interface(old_Center, "old_Center");                               
}
#pragma omp single
{
export_multidimentional_float_vector(old_Center, "old_Center");                                  
}
}
std::chrono::time_point<std::chrono::system_clock> optimization_benchmark = init_benchmark();                    
for (int iter_counter = 1; norm_iter_conv > THR_KMEANS; iter_counter += 1)                                       
{
std::chrono::duration<double> bench_res =
bench_loop(prev_loop_benchmark, loop_benchmark = init_benchmark());                              
#pragma omp parallel num_threads(3)
{
#pragma omp master
{
kmeans_progress(iter_counter, iter_conv, norm_iter_conv, bench_res, VERBOSITY);          
}
#pragma omp single
{
track_kmeans_progress(iter_counter, norm_iter_conv);                                     
}
#pragma omp single
{
compute_classes(Vec, old_Center, Classes);                                               
}
}
#pragma omp parallel num_threads(3)
{
#pragma omp master
{
optimize_center(Vec, new_Center, Classes);                                               
}
#pragma omp single
{
multidimentional_int_array_interface(Classes, "Classes");                                
}
#pragma omp single
{
export_multidimentional_integer_array(Classes, "Classes" + std::to_string(iter_counter));
}
}
long double prev_iter_conv = iter_conv;                                                                  
#pragma omp parallel num_threads(3)
{
#pragma omp master
{
iter_conv = convergence(new_Center, old_Center);                                         
norm_iter_conv = normalize_convergence(iter_conv, prev_iter_conv, iter_counter);         
}
#pragma omp single
{
multidimentional_float_vector_interface(new_Center, "new_Center");                       
}
#pragma omp single
{
export_multidimentional_float_vector(new_Center, "new_Center" + 
std::to_string(iter_counter));                                                   
}
}
old_Center.swap(new_Center);                                                                             
multidimentional_float_vector_interface(old_Center, "old_Center");                                       
}
std::chrono::time_point<std::chrono::system_clock> end = terminate_bench();                                      
std::pair<std::chrono::duration<double>, std::chrono::duration<double>> bench_res =
benchmark_results(start, optimization_benchmark, end);                                                   
kmeans_termination(bench_res, norm_iter_conv);                                                                   
export_kmeans_progress("kmeans_results");                                                                        
return 0;
}
