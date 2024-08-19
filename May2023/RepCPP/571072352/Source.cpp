#include <iostream>
#include <locale>
#include <omp.h>

#define EXP_NUM 30

using namespace std;

template <typename t>
void fillarray(t*& arr, int& size)
{
#pragma omp parallel for
for (int i = 0; i < size; i++)
arr[i] = sin(i + 1) + cos(i + 3);
}

template <typename t>
double bublesort(t* array, int& size)
{
double t_start = omp_get_wtime();
#pragma omp parallel for
for (int i = 0; i < size; i++)
{
for (int j = 0; j < size - 1; j++)
{
if (array[j] > array[j + 1])
swap(array[j], array[j + 1]);
}
}
return omp_get_wtime() - t_start;
}

template <typename t>
double oddEvenSortingV2(t* array, int& size)
{
double time_start = omp_get_wtime();

int res, i;
bool has_zero = false;

for (int i = 0; i < size; i += 2) {
res = 0;
#pragma omp parallel for private (i)
for (int j = 0; j < size - 1; j += 2) {
if (array[j] > array[j + 1]) {
swap(array[j], array[j + 1]);
res = 1;
}
}
#pragma omp parallel for private (i)
for (int j = 1; j < size - 1; j += 2)
if (array[j] > array[j + 1]) {
swap(array[j], array[j + 1]);
res = 1;
}
if (res == 0)
has_zero = true;
}

return omp_get_wtime() - time_start;
}

template <typename t>
double shellsort(t* array, int& size)
{
double t_start = omp_get_wtime();

for (int s = size / 2; s > 0; s /= 2)
{
#pragma omp parallel for
for (int i = s; i < size; i++)
for (int j = i - s; j >= 0 && array[j] > array[j + s]; j -= s)
swap(array[j], array[j + s]);
}

return omp_get_wtime() - t_start;
}

template <typename t>
double qsortRecursive(t* array, int size)
{
double t_start = omp_get_wtime();
long i = 0;
int j = size - 1;

t mid = array[size / 2];

do
{
while (array[i] < mid)
i++;
while (array[j] > mid)
j--;

if (i <= j)
{
swap(array[i], array[j]);
i++;
j--;
}
} while (i <= j);


#pragma omp task shared (array)
{
if (j > 0)
qsortRecursive(array, j + 1);
}
#pragma omp task shared (array)
{
if (size > i)
qsortRecursive(array + i, size - i);
}
#pragma omp taskwait

return omp_get_wtime() - t_start;
}

double AvgTrustedIntervalAVG(double*& times, int cnt)
{
double avg = 0.;
for (int i = 0; i < cnt; i++)
avg += times[i];
avg /= cnt;
double sd = 0., newAVg = 0.;
int newCnt = 0;
for (int i = 0; i < cnt; i++)
sd += (times[i] - avg) * (times[i] - avg);
sd /= (cnt - 1.0);
sd = sqrt(sd);

for (int i = 0; i < cnt; i++)
if (avg - sd <= times[i] && times[i] <= avg + sd)
{
newAVg += times[i];
newCnt++;
}
if (newCnt == 0) newCnt = 1;
return newAVg / newCnt;
}

int main()
{
setlocale(LC_ALL, "RUS");

double** times = new double* [8];

for (int j = 0; j < 8; j++)
{
times[j] = new double[EXP_NUM];
}

for (int i = 20000; i <= 50000; i += 10000)
{
int* arr_int = new int[i];
double* arr_double = new double[i];
fillarray(arr_int, i);
fillarray(arr_double, i);

for (int threads = 1; threads <= 4; threads++)
{
omp_set_num_threads(threads);
cout << "   " << omp_get_max_threads() << endl;
cout << "  " << i << endl;

for (int k = 0; k < EXP_NUM; k++)
{
if (threads == 1)
{
times[0][k] = bublesort(arr_int, i) * 1000;
times[1][k] = bublesort(arr_double, i) * 1000;
}
times[2][k] = oddEvenSortingV2(arr_int, i) * 1000;
times[3][k] = oddEvenSortingV2(arr_double, i) * 1000;

times[4][k] = shellsort(arr_int, i) * 1000;
times[5][k] = shellsort(arr_double, i) * 1000;

times[6][k] = qsortRecursive(arr_int, i) * 1000;
times[7][k] = qsortRecursive(arr_double, i) * 1000;
}
if (threads == 1)
{
cout << "  int () " << AvgTrustedIntervalAVG(times[0], 30) << endl;
cout << "  double () " << AvgTrustedIntervalAVG(times[1], 30) << endl;
}
cout << "  int (-) " << AvgTrustedIntervalAVG(times[2], 30) << endl;
cout << "  double (-) " << AvgTrustedIntervalAVG(times[3], 30) << endl;

cout << "  int " << AvgTrustedIntervalAVG(times[4], 30) << endl;
cout << "  double " << AvgTrustedIntervalAVG(times[5], 30) << endl;
cout << " quicksort int " << AvgTrustedIntervalAVG(times[6], 30) << endl;
cout << " quicksort double " << AvgTrustedIntervalAVG(times[7], 30) << endl;
}
delete[] arr_int;
delete[] arr_double;
}
}