#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <sys\timeb.h> 

#define N  14
#define NUM_OF_THREADS 7

int put(int Queens[], int row, int column, int* solutions)
{
int i;
for (i = 0; i < row; i++) {
if (Queens[i] == column || abs(Queens[i] - column) == (row - i))
return -1;
}
Queens[row] = column;

if (row == N - 1) {
(*solutions)++;
}
else {
for (i = 0; i < N; i++) { 
put(Queens, row + 1, i, solutions);
}
}
return 0;
}


int solve(int Queens[]) {
int i, localSolutions[NUM_OF_THREADS] = {0}, localQueens[NUM_OF_THREADS][N], solutions = 0;

#pragma omp parallel num_threads(NUM_OF_THREADS) shared(localSolutions, localQueens) private(i)
#pragma omp for schedule(dynamic)
for (i = 0; i < N; i++) {
put(localQueens[omp_get_thread_num()], 0, i, &localSolutions[omp_get_thread_num()]);
}

for (i = 0; i < NUM_OF_THREADS; i++)
solutions += localSolutions[i];

return solutions;
}



int main()
{
int Queens[N], diff, solutions;
struct timeb start, end;

ftime(&start);
solutions = solve(Queens);
ftime(&end);

diff = (int)(1000.0 * (end.time - start.time)
+ (end.millitm - start.millitm));

printf("Kimia Khabiri: 810196606 - Parsa Hoseininejad: 810196604\n");
printf("# solutions %d time: %u \n", solutions, diff);

return 0;

}