#include <limits>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

using namespace std;

void define_array(int array[6][8])
{
for (int i = 0; i < 6; i++)
{
for (int j = 0; j < 8; j++)
{
int value = rand() / 10;
array[i][j] = value;
printf("%d ", value);
}
printf("\n");
}
printf("\n");
}

float average(int array[6][8]){

float sum = 0;
int cnt = 0;

for (int i = 0; i < 6; i++) 
{
for (int j = 0; j < 8; j++)
{
sum += array[i][j];
cnt += 1;
}
}
return sum/cnt;
}

int max(int array[6][8]){

int max = numeric_limits<int>::min();

for (int i = 0; i < 6; i++) 
{
for (int j = 0; j < 8; j++)
{
if (array[i][j] > max)
max = array[i][j];
}
}
return max;
}

int min(int array[6][8]){

int min = numeric_limits<int>::max();

for (int i = 0; i < 6; i++) 
{
for (int j = 0; j < 8; j++)
{
if (array[i][j] < min)
min = array[i][j];
}
}
return min;
}

int get_multiple_of_three_cnt(int array[6][8]){

int cnt = 0;
for (int i = 0; i < 6; i++) {
for (int j = 0; j < 8; j++) {
if (array[i][j] % 3 == 0) {
cnt += 1;
}
}
}
return cnt;
}

int main() {

int d[6][8];
srand(time(NULL));
define_array(d);

#pragma omp parallel sections
{
#pragma omp section
{
printf("Averige value = %.3f, thread num %d\n", average(d), omp_get_thread_num());
}
#pragma omp section
{
printf("Min value = %d, max value = %d, thread num %d\n", min(d), max(d), omp_get_thread_num());
}
#pragma omp section
{
printf("Multiples of 3 count = %d, thread num %d\n", get_multiple_of_three_cnt(d), omp_get_thread_num());
}
}
}
