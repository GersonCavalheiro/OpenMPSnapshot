



#include <omp.h>
#include <iostream>
#include <math.h>


using namespace std;

uint16_t inline mpow2(uint16_t shftAmt) {
return (1 << shftAmt);
}

int main()
{
cout << omp_get_num_threads() << endl << omp_get_max_threads();
omp_set_num_threads(4);
cout << omp_get_num_threads() << endl << omp_get_max_threads();
uint64_t g;
uint64_t e;
uint64_t p;

uint16_t resultArray[4] = { 0, 0, 0, 0 };
uint64_t result = 0;

cout << "Enter g: ";
cin >> g;
cout << "Enter e: ";
cin >> e;
cout << "Enter p: ";
cin >> p;

double startTime, endTime, ticks;

uint16_t shiftArray[4] = { 0, 0, 0, 0 };
shiftArray[0] = e;
for (int i = 1; i < sizeof(shiftArray) / sizeof(shiftArray[0]); i++)
{
shiftArray[i] = e >> 16 * i;
}
int index = -1;
for (int i = 0; i < sizeof(shiftArray) / sizeof(shiftArray[0]); i++)
{
if (shiftArray[i] == 0)
{
int j = i + 1;
for (; j < sizeof(shiftArray) / sizeof(shiftArray[0]); j++)
{
if (shiftArray[j] == 0)
continue;
break;
}
if (j == sizeof(shiftArray) / sizeof(shiftArray[0]))
{
index = i;
break;
}
}
}

auto r0 = resultArray[0];
auto r1 = resultArray[1];
auto r2 = resultArray[2];
auto r3 = resultArray[3];
startTime = omp_get_wtime();






#pragma omp parallel sections private(p,g)
{
#pragma omp section 
{
if (0 >= index)
{
r0 = 0;
}
else
{
r0 = (int64_t)pow(g, mpow2(0)) % p;
r0 = (int64_t)pow(r0, r0) % p;
}
}
#pragma omp section
{
if (1 >= index)
{
r1 = 0;
}
else
{
r1 = (int64_t)pow(g, mpow2(1)) % p;
r1 = (int64_t)pow(r1, r1) % p;
}
}
#pragma omp section 
{
if (2 >= index)
{
r2 = 0;
}
else
{
r2 = (int64_t)pow(g, mpow2(2)) % p;
r2 = (int64_t)pow(r2, r2) % p;
}
}
#pragma omp section 
{
if (3 >= index)
{
r3 = 0;
}
else
{
r3 = (int64_t)pow(g, mpow2(3)) % p;
r3 = (int64_t)pow(r3, r3) % p;
}
}
}

endTime = omp_get_wtime();
ticks = omp_get_wtick();

for (int i = sizeof(resultArray) / sizeof(resultArray[0]) - 1; i >= 0; i--)
{
result += (uint64_t)resultArray[i] << (16 * i);
}

result = result % p;
cout << "Result: " << result << "\n";
cout << "Duration: " << endTime - startTime << "\n";
cout << "Timer accuracy: " << ticks << "\n";
}