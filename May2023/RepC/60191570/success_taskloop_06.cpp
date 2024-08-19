#include <assert.h>
void loop_1(int *v, int start, int end, int step, int offset, int chunk)
{
#pragma omp taskloop num_tasks(chunk)
for (int i = start; i < end; i += step) {
v[i+offset]++;
}
}
void loop_2(int *v, int start, int end, int step, int offset, int chunk)
{
#pragma omp taskloop num_tasks(chunk)
for (int i = start; i > end; i += step) {
v[offset-i]++;
}
}
void check_results(int *v, int start, int end, int step, int chunk_values)
{
for (int i = start; i < end; i += step)
assert(v[i] == chunk_values);
}
void init(int *v, int size)
{
for(int i = 0; i < size; ++i)
v[i] = 0;
}
int main(int argc, char *argv[]) {
const int n = 2000;
int v[n];
{
int start  =     0;
int end    =  2000;
int step   =    10;
int offset =     0;
int chunk_values = 20;
init(v, n);
for (int i = 1; i <= chunk_values; i++)
loop_1(v, start, end, step, offset, i);
check_results(v, 0, n, step, 20);
}
{
int start  =  2000;
int end    =     0;
int step   =   -10;
int offset =  2000;
int chunk_values = 20;
init(v, n);
for (int i = 1; i <= chunk_values; i++)
loop_2(v, start, end, step, offset, i);
check_results(v, 0, n, -step, 20);
}
{
int start  = -2000;
int end    =     0;
int step   =    10;
int offset =  2000;
int chunk_values = 20;
init(v, n);
for (int i = 1; i <= chunk_values; i++)
loop_1(v, start, end, step, offset, i);
check_results(v, 0, n, step, 20);
}
{
int start  =     0;
int end    = -2000;
int step   =   -10;
int offset =     0;
int chunk_values = 20;
init(v, n);
for (int i = 1; i <= chunk_values; i++)
loop_2(v, start, end, step, offset, i);
check_results(v, 0, n, -step, 20);
}
{
int start  = -1000;
int end    =  1000;
int step   =    10;
int offset =  1000;
int chunk_values = 20;
init(v, n);
for (int i = 1; i <= chunk_values; i++)
loop_1(v, start, end, step, offset, i);
check_results(v, 0, n, step, 20);
}
{
int start  =  1000;
int end    = -1000;
int step   =   -10;
int offset =  1000;
int chunk_values = 20;
init(v, n);
for (int i = 1; i <= chunk_values; i++)
loop_2(v, start, end, step, offset, 10);
check_results(v, 0, n, -step, chunk_values);
}
return 0;
}
