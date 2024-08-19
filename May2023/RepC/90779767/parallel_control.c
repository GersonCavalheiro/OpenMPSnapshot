#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <string.h>
int n;
int n_max;
int n_threads;
unsigned char* primes;
int main (int argc, char **argv) {
omp_lock_t index_lock;
if(argc <  3) {
printf("Usage: %s n [--print all]\n"
"n: maximum number\n"
"n_threads: number of threads\n"
"--print: print num of primes (optional)\n"
"all: print all primes (optional)\n", argv[0]);
exit(1);
}
n_max = atoi(argv[1]);
n_threads = atoi(argv[2]);
primes = (unsigned char*)malloc(sizeof(unsigned char) * (n_max+1));
for (int i = 0; i <= n_max; i++) {
primes[i] = 1;
}
primes[0] = 0;
primes[1] = 1;
omp_init_lock(&index_lock);
n = 2;
int nthreads = omp_get_num_threads();
#pragma omp parallel num_threads(n_threads)
while(1) {
omp_set_lock(&index_lock);
if(n > sqrt(n_max)) {
omp_unset_lock(&index_lock);
break;
}
int i = n;
n++;
omp_unset_lock(&index_lock);
if (primes[i] == 1) {
for (int j = 2; i*j <= n_max; j++) {
primes[i*j] = 0;
}
}
nthreads = omp_get_num_threads();
}
if (argc > 3) {
if(!strcmp(argv[3], "--print") || !strcmp(argv[3], "--print\n")) {
int count = 0;
if((argc > 4) && (!strcmp(argv[4], "all") || !strcmp(argv[4], "all\n"))) {
for (int i = 0; i <= n_max; i++) {
if (primes[i] == 1) {
printf("%d\n", i);
count++;
}
}
} else {
for (int i = 0; i <= n_max; i++) {
if (primes[i] == 1) {
count++;
}
}
}
printf("There are %d primes less than or equal to %d\n\n", count, n_max);
printf("Number of threads = %d\n", nthreads);
}
}
return 0;
}
