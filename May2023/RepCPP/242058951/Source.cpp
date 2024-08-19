#include<iostream>
#include<omp.h>
#include<ctime>
using namespace std;

clock_t measure(int num, clock_t clk) {
omp_set_num_threads(num);
#pragma omp parallel for
for (int i = 0; i < 8; i++) {
int id = omp_get_thread_num();
printf("iteration # = %d ---- thread # %d\n", i, id);
}
return clock() - clk;
}

int main() {

clock_t clk1 = clock();
clk1 = measure(1, clk1);
cout << endl;

clock_t clk2 = clock();
clk2 = measure(2, clk2);
cout << endl;

clock_t clk4 = clock();
clk4 = measure(4, clk4);
cout << endl;

clock_t clk8 = clock();
clk8 = measure(8, clk8);
cout << endl;

cout << endl;
cout << "Running time for 1 thread: " << clk1 << endl;
cout << "Running time for 2 threads: " << clk2 << endl;
cout << "Running time for 4 threads: " << clk4 << endl;
cout << "Running time for 8 threads: " << clk8 << endl;
getchar();
}