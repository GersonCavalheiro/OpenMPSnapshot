#include <omp.h>
#include <stdio.h>

int main(int argc, char** argv) {

omp_lock_t lockA;
omp_lock_t lockB;
omp_init_lock(&lockA);
omp_init_lock(&lockB);

size_t T=8;
int t_hello[T];
int i=0;
for(i=0;i<T;i++) {
t_hello[i]=-1;
}
i=0;
#pragma omp parallel num_threads(8)
{
omp_set_lock(&lockA);
t_hello[i]=omp_get_thread_num();
i++;
omp_unset_lock(&lockA);
}

for(i=0;i<T;i++) {
omp_set_lock(&lockB);
printf("Hello from %d\n",t_hello[i]);
omp_unset_lock(&lockB);
}

return 0;
}
