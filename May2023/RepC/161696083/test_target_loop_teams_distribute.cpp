#include <omp.h>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <vector>
#include "ompvv.h"
template <typename T> void saxpy_d(T *x, T *y, T a, int n) {
#pragma omp target teams distribute parallel for map(tofrom: x [0:n]) map(to: y [0:n])
for (int i = 0; i < n; ++i)
x[i] = a * x[i] + y[i];
}
template <typename T> void saxpy_l(T *x, T *y, T a, int n) {
#pragma omp target teams loop map(tofrom : x [0:n]) map(to : y [0:n])
for (int i = 0; i < n; ++i)
x[i] = a * x[i] + y[i];
}
template <typename T> int verify(T *x, T v, int n) {
for (int i = 0; i < n; ++i)
if (x[i] != v)
return 0;
return 1;
}
int test_target_loop_teams_distribute() {
OMPVV_INFOMSG("test_target_loop_teams_distribute");
srand(time(NULL));
int n = (rand() % 4096) + 4096, errors = 0, tmp = 0;
int a = (rand() % 1024) + 1, x = rand() % 2048, y = rand() % 4096;
std::vector<int> xd(n, x), xl(n, x), yg(n, y);
saxpy_d(xd.data(), yg.data(), a, n);
saxpy_l(xl.data(), yg.data(), a, n);
tmp = verify(xd.data(), a * x + y, n) + verify(xl.data(), a * x + y, n) * 2;
OMPVV_TEST_AND_SET_VERBOSE(errors, 3 != tmp);
return errors;
}
int main(int argc, char **argv) {
int errors = 0;
OMPVV_TEST_OFFLOADING;
OMPVV_TEST_AND_SET_VERBOSE(errors, test_target_loop_teams_distribute());
OMPVV_REPORT_AND_RETURN(errors);
}
