#include <iostream>
#include <omp.h>
#include "Stopwatch.h"
#include <mutex>

using namespace std;

static long long sum(const long long n) {
return n*(n + 1)/2;
}

long long sumSerial(const int n) {
long long sum = 0;
for (int i=1; i <= n; i++) {
sum += i;
}
return sum;
}

static long long sumPar1(const int n) {
long long sum = 0;
#pragma omp parallel for default(none) shared(sum)
for (int i = 0; i <= n; i++)
{
#pragma omp critical
sum += i;
}
return sum;
}

static long long sumPar2(const int n) {
long long sum = 0;
#pragma omp parallel for default(none) shared(sum)
for (int i = 0; i <= n; i++)
{
#pragma omp atomic
sum += i;
}

return sum;
}

static long long sumPar3(const int n) {
long long sum = 0;
#pragma omp parallel for default(none) shared(sum) reduction(+:sum)
for (int i = 0; i <= n; i++)
{
sum += i;
}
return sum;
}

static long long sumPar4(const int n) {
long long sum = 0;
bool m_writeLocked = false;
mutex m_mutex;
condition_variable m_writingAllowed;

#pragma omp parallel for default(none) shared(sum, m_writeLocked, m_mutex, m_writingAllowed)
for (int i = 0; i <= n; i++)
{
unique_lock<mutex> monitor(m_mutex);
while (m_writeLocked) {
m_writingAllowed.wait(monitor);
}
m_writeLocked = true;
sum += i;
m_writeLocked = false;
}
return sum;
}

void summation() {
cout << "\nSummation Tests" << endl;

const int64_t N = 10000000;
Stopwatch sw;

sw.Start();
int64_t sum0 = sum(N);
sw.Stop();
cout << "Explicit: " << sum0 << " in " << sw.GetElapsedTimeMilliseconds() << " ms" << endl << endl;

sw.Start();
int64_t sumS = sumSerial(N);
sw.Stop();
cout << "Serial: " << sumS << " in " << sw.GetElapsedTimeMilliseconds() << " ms" << endl;
cout << boolalpha << "The two operations produce the same results: " << (sumS == sum0) << endl << endl;

sw.Start();
int64_t sum1 = sumPar1(N);
sw.Stop();
cout << "Critical section: " << sum1 << " in " << sw.GetElapsedTimeMilliseconds() << " ms" << endl;
cout << boolalpha << "The two operations produce the same results: " << (sum1 == sum0) << endl << endl;

sw.Start();
int64_t sum2 = sumPar2(N);
sw.Stop();
cout << "Atomic access: " << sum2 << " in " << sw.GetElapsedTimeMilliseconds() << " ms" << endl;
cout << boolalpha << "The two operations produce the same results: " << (sum2 == sum0) << endl << endl;

sw.Start();
int64_t sum3 = sumPar3(N);
sw.Stop();
cout << "Reduction: " << sum3 << " in " << sw.GetElapsedTimeMilliseconds() << " ms" << endl;
cout << boolalpha << "The two operations produce the same results: " << (sum3 == sum0) << endl << endl;

sw.Start();
int64_t sum4 = sumPar4(N);
sw.Stop();
cout << "Explicit locks: " << sum4 << " in " << sw.GetElapsedTimeMilliseconds() << " ms" << endl;
cout << boolalpha << "The two operations produce the same results: " << (sum4 == sum0) << endl << endl;

}
