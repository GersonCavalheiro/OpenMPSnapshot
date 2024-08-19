#include <stdio.h>
#include <assert.h>
#include <unistd.h>
#include <stdlib.h>
void generate0(int length, int *a, int val)
{
#pragma omp task out( { a[k], k = 0;length } )
{
usleep(50);
int i;
for (i = 0; i < length; i++)
{
a[i] = val;
}
}
}
void consume0(int length, int *a, int val)
{
#pragma omp task in( { a[k], k = 0;length } )
{
usleep(50);
int i;
for (i = 0; i < length; i++)
{
if (a[i] != val)
{
fprintf(stderr, "a[%d] == %d but should be %d\n", i, a[i], val);
abort();
}
}
}
}
void generate1(int start, int length, int *a, int val)
{
#pragma omp task out( { a[k], k = start;length } )
{
usleep(50);
int i;
for (i = start; i < (start + length); i++)
{
a[i] = val;
}
}
}
void consume1(int start, int length, int *a, int val)
{
#pragma omp task in( { a[k], k = start;length } )
{
usleep(50);
int i;
for (i = start; i < (start + length); i++)
{
if (a[i] != val)
{
fprintf(stderr, "a[%d] == %d but should be %d\n", i, a[i], val);
abort();
}
}
}
}
enum { NUM_ITEMS = 100, NUM_TASKS = 10000 };
int vec[NUM_ITEMS];
int main(int argc, char *argv[])
{
fprintf(stderr, "Initializing %d items...\n", NUM_ITEMS);
int i;
for (i = 0; i < NUM_ITEMS; i++)
{
vec[i] = i;
}
fprintf(stderr, "First round: Creating %d tasks...\n", NUM_TASKS);
for (i = 0; i < NUM_TASKS; i++)
{
int start = (i % NUM_ITEMS);
int length = 20;
if ((start + length) >= NUM_ITEMS)
{
length = NUM_ITEMS - start;
}
int val = i;
generate0(length, vec + start, val);
consume0(length, vec + start, val);
}
fprintf(stderr, "Second round: Creating %d tasks...\n", NUM_TASKS);
for (i = 0; i < NUM_TASKS; i++)
{
int start = (i % NUM_ITEMS);
int length = 20;
if ((start + length) >= NUM_ITEMS)
{
length = NUM_ITEMS - start;
}
int val = i;
generate1(start, length, vec, val);
consume1(start, length, vec, val);
}
fprintf(stderr, "Waiting tasks...\n");
#pragma omp taskwait
return 0;
}
