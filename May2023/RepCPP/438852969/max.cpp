
#include "max.h"


int Max::get_max(int array[], int array_length, int threads) {
auto start = chrono::system_clock::now();
int max_el = array[0];
if (debug) {
cout << "Start of searching max element of array\n";
}
if (omp) {
if (debug) {
cout << "OMP option is selected\n";
}
#pragma omp parallel for shared(array, max_el, array_length, debug) default(none) num_threads(threads)
for (int i = 1; i < array_length; i++) {
int tid = omp_get_thread_num();
if (debug) {
printf("Iter №%d:\n\t"
"   tid: %d\n\t"
"   arr element: %d\n\t"
"   max_el: %d\n", i, tid, array[i], max_el);
}
if (array[i] >= max_el) {
max_el = array[i];
if (debug) {
printf("Iter №%d:\n\t"
"   tid: %d\n\t"
"   new max_el: %d\n", i, tid, max_el);
}
}
}
} else {
for (int i = 1; i < array_length; i++) {
int tid = 0;
if (debug) {
printf("Iter №%d:\n\t"
"   tid: %d\n\t"
"   arr element: %d\n\t"
"   max_el: %d\n", i, tid, array[i], max_el);
}
if (array[i] >= max_el) {
max_el = array[i];
if (debug) {
printf("Iter №%d:\n\t"
"   tid: %d\n\t"
"   new max_el: %d\n", i, tid, max_el);
}
}
}
}
auto end = chrono::system_clock::now();
long int total_time = chrono::duration_cast<chrono::milliseconds>(end - start).count();
printf("Max element of array: %d\nExecution time: %ld ms\n", max_el, total_time);
if (log_time) {
ofstream execution_time_log;
execution_time_log.open("execution_time_log.csv", fstream::app);
execution_time_log << omp << ", " << total_time << endl;
execution_time_log.close();
}
return max_el;
}
