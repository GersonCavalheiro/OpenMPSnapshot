#include<stdlib.h>
#include<stdio.h>
#include<omp.h>
int main(){
#ifdef   _OPENMP
printf("\nKompilator rozpoznaje dyrektywy OpenMP\n");
#endif
static int f_threadprivate = 101;
int liczba_watkow = 33;
int a_shared = 1;
int b_private = 2;
int c_firstprivate = 3;
int e_atomic = 5;
#pragma omp threadprivate(f_threadprivate)
printf("przed wejsciem do obszaru rownoleglego -  nr_threads %d, thread ID %d\n", omp_get_num_threads(), omp_get_thread_num());
printf("\ta_shared \t= %d\n", a_shared);
printf("\tb_private \t= %d\n", b_private);
printf("\tc_firstprivate \t= %d\n", c_firstprivate);
printf("\te_atomic \t= %d\n", e_atomic);
printf("\tf_threadprivate = %d\n", f_threadprivate);
#pragma omp parallel default(none) shared(a_shared, e_atomic) private(b_private) firstprivate(c_firstprivate) num_threads(7)
{
int i;
int d_local_private;
d_local_private = a_shared + c_firstprivate;
#pragma omp barrier
for(i=0;i<10;i++){
#pragma omp critical(a_shared)
a_shared ++;
}
for(i=0;i<10;i++){
c_firstprivate += omp_get_thread_num();
}
for(i=0;i<10;i++){
e_atomic += omp_get_thread_num();
}
f_threadprivate = omp_get_thread_num();
#pragma omp critical
{
printf("\nw obszarze równoległym: aktualna liczba watkow %d, moj ID %d\n", omp_get_num_threads(), omp_get_thread_num());
printf("\ta_shared \t= %d\n", a_shared);
printf("\tb_private \t= %d\n", b_private);
printf("\tc_firstprivate \t= %d\n", c_firstprivate);
printf("\td_local_private = %d\n", d_local_private);
printf("\te_atomic \t= %d\n", e_atomic);
printf("\tf_threadprivate = %d\n", f_threadprivate);
}
}
#pragma omp parallel num_threads(10)
{
#pragma omp critical
printf("\nw obszarze równoległym: aktualna liczba watkow %d, moj ID %d\n", omp_get_num_threads(), omp_get_thread_num());
printf("\tf_threadprivate = %d\n", f_threadprivate);
}
printf("po zakonczeniu obszaru rownoleglego:\n");
printf("\ta_shared \t= %d\n", a_shared);
printf("\tb_private \t= %d\n", b_private);
printf("\tc_firstprivate \t= %d\n", c_firstprivate);
printf("\te_atomic \t= %d\n", e_atomic);
printf("\tf_threadprivate = %d\n", f_threadprivate);
}
