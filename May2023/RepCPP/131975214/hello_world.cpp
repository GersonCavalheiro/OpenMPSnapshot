#include <iostream>
#include <omp.h>
#include <stdio.h>

using namespace std;

int main(int argc, char const *argv[]) {

printf("Maximum number of threads: %d\n", omp_get_max_threads());

int parallel = omp_in_parallel() ? 1 : 0;
printf("Am I in a parallel construct?\n");
if (parallel) cout << "Yes\n"; else cout << "No\n";
if (parallel) cout << "Thread: " << omp_get_thread_num() << "\n";

#pragma omp parallel for
for (int i = 0; i < omp_get_max_threads(); i++) {
parallel = omp_in_parallel() ? 1 : 0;
printf("Am I in a parallel construct?\n");
if (parallel) cout << "Yes\n"; else cout << "No\n";
if (parallel) cout << "Thread: " << omp_get_thread_num() << "\n";
cout << "Testing openMP!" << '\n';
}

return 0;
}
