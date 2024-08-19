#include<stdio.h>
#include<stdlib.h>
#include<omp.h>
#include <sys/time.h>
#define SEED_DEFAULT 100
#define TAM_DEFAULT 200000000
long int gettime_interval(struct timeval tv1, struct timeval tv2) {
return (tv2.tv_sec - tv1.tv_sec) * 1000000 + (tv2.tv_usec - tv1.tv_usec);
}
int main(int argc, char* argv[]) {
int tam, i,nthreads, tid;
unsigned long long sum;
int *vec1, *vec2;
int seed;
struct timeval tv[2];
if(argc == 2) {
tam = atoi(argv[1]);
seed = SEED_DEFAULT;
} else if(argc >= 3) {
tam = atoi(argv[1]);
seed = atoi(argv[2]);
} else {
tam = TAM_DEFAULT;
seed = SEED_DEFAULT;
}
srand(seed);
vec1 = (int*) malloc(tam*sizeof(int));
vec2 = (int*) malloc(tam*sizeof(int));
omp_set_num_threads(1);
gettimeofday(&tv[0],NULL);
#pragma omp parallel for schedule(static) shared(vec1,vec2)
for(i=0;i<tam;i++) {
vec1[i] =  i%10000; 
vec2[i] =  i%10000;
}
sum = 0;   
#pragma omp parallel for reduction(+:sum) schedule(static) shared(vec1,vec2)
for(i=0;i<tam;i++) {
sum += vec1[i] * vec2[i];
}
if(tid == 0) {
printf("O resultado da operacao eh: %lld\n", sum);
}
gettimeofday(&tv[1],NULL);
printf("\n%ld microsegundos\n\n", gettime_interval(tv[0],tv[1]));
free(vec1);
free(vec2);
return 0;
}
