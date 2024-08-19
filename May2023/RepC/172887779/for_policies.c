#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include "../omp_logs.h"
struct data{
int x;
int* sum;
};
void sum(void* d) {
struct data* data = (struct data*) d;
*(data->sum) += data->x;
}
int main(int argc, char** argv) {
if (argc != 2) {
fprintf(stderr, "Usage %s N\n", argv[0]);
return 1;
}
int N = atoi(argv[1]);
task_list* l = task_list_init();
int s = 0;
#pragma omp parallel for schedule(static) reduction (+:s)
for (int j = 0; j < N; j++) {
struct data d = {j, &s};
log_task(&l, "Sum", j, omp_get_thread_num(), sum, (void*) &d);
}
tasks_to_svg(l, "for_static.svg", 1);
l = task_list_init();
s = 0;
#pragma omp parallel for schedule(dynamic) reduction (+:s)
for (int j = 0; j < N; j++) {
struct data d = {j, &s};
log_task(&l, "Sum", j, omp_get_thread_num(), sum, (void*) &d);
}
tasks_to_svg(l, "for_dynamic.svg", 1);
l = task_list_init();
s = 0;
#pragma omp parallel for schedule(guided) reduction (+:s)
for (int j = 0; j < N; j++) {
struct data d = {j, &s};
log_task(&l, "Sum", j, omp_get_thread_num(), sum, (void*) &d);
}
tasks_to_svg(l, "for_guided.svg", 1);
return 0;
}
