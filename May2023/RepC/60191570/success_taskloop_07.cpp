template<typename E>
void foo() {
#pragma omp taskloop num_tasks (5)
for (E i = 0 ; i <10; ++i) {
}
}
