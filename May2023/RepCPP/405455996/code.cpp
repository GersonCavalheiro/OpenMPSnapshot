#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <Windows.h>

#define N  14

int solutions;

int put(int Queens[], int row, int column)
{
int i;

for (i = 0; i < row; i++) {
if (Queens[i] == column || abs(Queens[i] - column) == (row - i))
return -1;
}

Queens[row] = column;
if (row == N - 1) {
#pragma omp critical
solutions++;
}
else {

for (i = 0; i < N; i++) { 
put(Queens, row + 1, i);
}

}
return 0;
}


void solve() {
#pragma omp parallel
#pragma omp single  
{

for (int i = 0; i < N; i++) {
#pragma omp task
put(new int[N], 0, i);
}
}

}


int main()
{
time_t t0 = 0, tf = 0, t0s = 0, tfs = 0;
DWORD starttime, elapsedtime;

t0 = timeGetTime();

solve();

tf = timeGetTime();

printf("Hossein Soltanloo - 810195407\n");
printf("# solutions %d time: %10d ms \n", solutions, (int)(tf - t0));

return 0;

}