#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <assert.h>

long long int sum(int *v,long long int n){
long long int i,sum=0;
int myid, nthreads;

#pragma omp parallel private(i, myid) shared(sum, nthreads)
{
myid = omp_get_thread_num();
#pragma omp single
nthreads = omp_get_num_threads();

for(i = myid; i < n; i += nthreads)
#pragma omp critical
sum+=v[i];
}
return sum;
}


int main(int argc, char **argv){
char *awnser[] = { "bad", "ok" };

long long int i, vsum, n;
int *v;
double elapsed, start, end;

if(argc != 2){
fprintf(stderr, "Usage: %s <number of elements>\n", argv[0]);
exit(EXIT_FAILURE);
}

n = atoll(argv[1]);
printf("number of elements: %lld\n", n);

v = (int *) malloc(n * sizeof(int));
assert(v != NULL);

for(i = 0; i < n; i++)
v[i] = 1; 

start = omp_get_wtime();
vsum = sum(v, n);
end = omp_get_wtime();

elapsed = end - start;

printf("sum value is %s\ntime: %.3f seconds\n", awnser[vsum == n], elapsed);

free(v);

return 0;
}
