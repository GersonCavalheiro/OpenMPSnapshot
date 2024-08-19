#include "omp.h"

void print_omp(experiment_result_t* results) {
printf("\nOpenMP\n");

run_experiments(results, integrate_seq);
print_experiment_result("seq", results);

run_experiments(results, integrate_omp_fs);
print_experiment_result("omp_fs", results);

run_experiments(results, integrate_omp);
print_experiment_result("omp", results);

run_experiments(results, integrate_omp_reduce);
print_experiment_result("omp_reduce", results);

run_experiments(results, integrate_omp_reduce_dyn);
print_experiment_result("omp_reduce_dyn", results);

run_experiments(results, integrate_omp_atomic);
print_experiment_result("omp_atomic", results);


run_experiments(results, integrate_omp_cs);
print_experiment_result("omp_cs", results);

run_experiments(results, integrate_omp_mtx);
print_experiment_result("omp_mtx", results);
}

void print_cpp(experiment_result_t* results) {
printf("\nCPP\n");

run_experiments(results, integrate_cpp);
print_experiment_result("cpp", results);

run_experiments(results, integrate_cpp_cs);
print_experiment_result("cpp_cs", results);

run_experiments(results, integrate_cpp_atomic);
print_experiment_result("cpp_atomic", results);

#if !defined( __GNUC__) || (__GNUC__ > 10)
run_experiments(results, integrate_cpp_reduce_1);
print_experiment_result("cpp_reduce", results);
#endif

run_experiments(results, integrate_cpp_reduce_2);
print_experiment_result("cpp_reduce_barrier", results);
}

int main(int argc, char** argv)
{
experiment_result_t* results = (experiment_result_t*)aligned_alloc(CACHE_LINE, omp_get_num_procs() * sizeof(experiment_result_t));
#pragma warning(disable : 4996)
freopen("integral_output.txt", "w", stdout);
print_omp(results);
print_cpp(results);
fclose(stdout);

aligned_free(results);
return 0;
}