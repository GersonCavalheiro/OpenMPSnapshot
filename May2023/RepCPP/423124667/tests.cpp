#include "quicksort.hpp"
#include <algorithm>
#include <iostream>
#include <chrono>

#if defined(_OPENMP)
#include <omp.h>
#endif

using namespace std;
using namespace std::chrono;

string random_string(size_t len)
{
auto randchar = []() -> char
{
const char charset[] =
"0123456789"
"ABCDEFGHIJKLMNOPQRSTUVWXYZ"
"abcdefghijklmnopqrstuvwxyz";
const size_t max_index = (sizeof(charset) - 1);
return charset[ rand() % max_index ];
};
string str(len, 0);
generate_n(str.begin(), len, randchar);
return str;
}

int runtest_integers(size_t len) {
int arr[len];
unsigned int i;
for (i = 0; i < len; i++) {
arr[i] = rand()%100;
}
#pragma omp parallel
{
#pragma omp single
{
quicksort(arr, len);
}
}
return 0;
}

int runtest_strings(size_t len) {
string arr[len];
unsigned int i;
for (i = 0; i < len; i++) {
arr[i] = random_string(3);
}
#pragma omp parallel
{
#pragma omp single
{
quicksort(arr, len);
}
}
return 0;
}

int main() {
int n = 100000;

#if defined(_OPENMP)
cout << "number of abailable threads: " << omp_get_max_threads() << endl;
#endif

cout << "quick-sorting " << n << " integers: ";
auto start = high_resolution_clock::now();
auto status = runtest_integers(n);
auto end = high_resolution_clock::now();
auto duration = duration_cast<microseconds>(end - start);
if (status == 0) {
cout << "Success" << endl;
cout << "execution took " << duration.count() << " nanoseconds" << endl;
} else {
cout << "Failues" << endl;
}

cout << "quick-sorting " << n << " strings: ";
start = high_resolution_clock::now();
status = runtest_strings(n);
end = high_resolution_clock::now();
duration = duration_cast<microseconds>(end - start);
if (status == 0) {
cout << "Success" << endl;
cout << "execution took " << duration.count() << " nanoseconds" << endl;
} else {
cout << "Failues" << endl;
}

return 0;
}