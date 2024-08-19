#include <omp.h>
#include <stdio.h>

int main(int argc, char*argv[]) {
int i, t = get_omp_num_threads();
#pragma omp parallel for
for(i=0;i<t;i++) {
fprintf(stdout, "Thread #%d\n", i);
}
}
