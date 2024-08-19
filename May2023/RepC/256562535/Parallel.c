#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
void printAll(int32_t* a, int32_t* b, int n);
int main(int argc, char* argv[]) {
if (argc != 3) {
fprintf(stderr, "arg missing! ./Parallel n t\n");
return EXIT_FAILURE;
}
int n = atoi(argv[1]);
int t = atoi(argv[2]);
printf("%d\n", n);
int32_t* a = malloc(sizeof(int32_t) * n);
int32_t* b = malloc(sizeof(int32_t) * n);
b[0] = 0;
for (int i = 0; i < n; i++)
a[i] = i;
printf("using %ld MB\n", (sizeof(int32_t) * n * 2) / 1000000);
printAll(a, b, n);
int last[255];             
int connector[255] = {0};  
printf("Threads %d\n ", t);
#pragma omp parallel num_threads(t)
{
int t_nums = omp_get_num_threads();
int t_id = omp_get_thread_num();
int ops = n / t_nums;  
int offset = ops * t_id;
for (int i = offset; i < offset + ops; i++) {
if ((i % ops) != 0)  
b[i] = b[i - 1] + a[i - 1];
}
last[t_id] = b[offset + ops - 1] + a[offset + ops - 1];  
#pragma omp barrier
for (int i = 0; i < t_id; i++) {  
connector[t_id] += last[i];
}
#pragma omp barrier
for (int i = offset; i < offset + ops; i++)
b[i] += connector[t_id];
}
printAll(a, b, n);
free(b);
free(a);
}
void printAll(int32_t* a, int32_t* b, int n) {
if (n > 200) {
puts("skipping print");
printf("LAST VALUE:%d\n", b[n - 1]);
return;
}
puts("");
printf("b:");
for (int i = 0; i < n; i++)
printf(" %d", b[i]);
puts("");
printf("a:");
for (int i = 0; i < n; i++)
printf(" %d", a[i]);
puts("");
}
