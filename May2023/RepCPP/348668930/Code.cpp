#include <iostream>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <windows.h>

#define N 14

using namespace std;

int solutions;

void printGroupMemberNames() {
cout << "Group Members:" << endl;
cout << "1 - Aryan Haddadi 810196448" << endl;
cout << "2 - Iman Moradi 810196560" << endl;
cout << "-------------" << endl;
}

int put(int Queens[], int row, int column) {
int i;
for(i = 0; i < row; i++) {
if (abs(Queens[i] - column) == (row - i) || Queens[i] == column)
return -1;
}
Queens[row] = column; 
if(row == N - 1) {
#pragma omp atomic
solutions++;
}
else {
for(i = 0; i < N; i++) { 
put(Queens, row + 1, i);
}
}
return 0;
}


void solve() {
int i;
#pragma omp parallel for num_threads(N)
for (i = 0; i < N; i++) {
int* Queens = new int[N];
put(Queens, 0, i);
}
}



int main() {
printGroupMemberNames();

DWORD start, end;

start = timeGetTime();
solve();
end = timeGetTime();

printf("# solutions %d time: %f miliseconds.\n", solutions, difftime(end, start));

return 0;
}