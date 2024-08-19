#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <sys/time.h>
void rand_str(char *dest, size_t length) {
char charset[] = "0123456789";
while (length-- > 0) {
size_t index = (double) rand() / RAND_MAX * (sizeof charset - 1);
*dest++ = charset[index];
}
}
long algori8mos(int m, int n, int l, int threads) {
int** diffs = (int**) malloc( m * sizeof(int*));
for (int i = 0; i < m; i++)
diffs[i] = (int*) calloc(l, n * sizeof(int));
char** StringsA = (char**) malloc( m * sizeof(char*));
for (int i = 0; i < m; i++)
StringsA[i] = (char*) malloc( l * sizeof(char));
char** StringsB = (char**) malloc( n * sizeof(char*));
for (int i = 0; i < n; i++)
StringsB[i] = (char*) malloc( l * sizeof(char));
long totalDiff = 0;	
struct timeval start,finish;
double totalTime;
for(int i = 0; i < m; i++){
rand_str(StringsA[i], l);
}
for(int i = 0; i < n; i++){
rand_str(StringsB[i], l);
}
gettimeofday(&start, NULL);
#pragma omp parallel num_threads(threads)
{
long threadDiff = 0;
#pragma omp for schedule(static) collapse(3) ordered nowait
for(int i = 0; i < m; i++){
for(int j = 0; j < n; j++){
for(int pshfio = 0; pshfio < l; pshfio++){
if(StringsA[i][pshfio] != StringsB[j][pshfio]){
threadDiff++;
#pragma omp atomic
diffs[i][j]++;
}
}
}
}
#pragma omp atomic
totalDiff += threadDiff;
}
gettimeofday(&finish, NULL);
free(diffs);
free(StringsA);
free(StringsB);
totalTime = 
(double)(finish.tv_usec - start.tv_usec) / 1000.0L + 
(double)(finish.tv_sec - start.tv_sec) * 1000.0L;
printf("Total time: %f ms\n", totalTime);
return totalDiff;
}
int main(int argc, const char** argv) {
int m = atoi(argv[1]);
int n = atoi(argv[2]);
int l = atoi(argv[3]);
int threads = atoi(argv[4]);
srand (0);
puts("\nTask size: 1 character.");
printf("Total humming distance: %ld\n", algori8mos(m, n, l, threads));
return EXIT_SUCCESS;
}
