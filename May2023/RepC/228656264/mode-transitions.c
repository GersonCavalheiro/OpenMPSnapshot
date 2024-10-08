#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <math.h>
#include <openacc.h>
void t1()
{
int n = 0, arr[32], i;
for (i = 0; i < 32; i++)
arr[i] = 0;
#pragma acc parallel copy(n, arr) num_gangs(1) num_workers(1) vector_length(32)
{
int j;
n++;
#pragma acc loop vector
for (j = 0; j < 32; j++)
arr[j]++;
n++;
}
assert (n == 2);
for (i = 0; i < 32; i++)
assert (arr[i] == 1);
}
void t2()
{
int n[32], arr[1024], i;
for (i = 0; i < 1024; i++)
arr[i] = 0;
for (i = 0; i < 32; i++)
n[i] = 0;
#pragma acc parallel copy(n, arr) num_gangs(32) num_workers(1) vector_length(32)
{
int j, k;
#pragma acc loop gang(static:*)
for (j = 0; j < 32; j++)
n[j]++;
#pragma acc loop gang
for (j = 0; j < 32; j++)
#pragma acc loop vector
for (k = 0; k < 32; k++)
arr[j * 32 + k]++;
#pragma acc loop gang(static:*)
for (j = 0; j < 32; j++)
n[j]++;
}
for (i = 0; i < 32; i++)
assert (n[i] == 2);
for (i = 0; i < 1024; i++)
assert (arr[i] == 1);
}
void t4()
{
int n[32], arr[1024], i;
for (i = 0; i < 1024; i++)
arr[i] = i;
for (i = 0; i < 32; i++)
n[i] = 0;
#pragma acc parallel copy(n, arr) num_gangs(32) num_workers(1) vector_length(32)
{
int j, k;
#pragma acc loop gang(static:*)
for (j = 0; j < 32; j++)
n[j]++;
#pragma acc loop gang
for (j = 0; j < 32; j++)
{
#pragma acc loop vector
for (k = 0; k < 32; k++)
if ((arr[j * 32 + k] % 2) != 0)
arr[j * 32 + k] *= 2;
}
#pragma acc loop gang(static:*)
for (j = 0; j < 32; j++)
n[j]++;
}
for (i = 0; i < 32; i++)
assert (n[i] == 2);
for (i = 0; i < 1024; i++)
assert (arr[i] == ((i % 2) == 0 ? i : i * 2));
}
void t5()
{
int n[32], arr[1024], i;
for (i = 0; i < 1024; i++)
arr[i] = i;
for (i = 0; i < 32; i++)
n[i] = 0;
#pragma acc parallel copy(n, arr) num_gangs(32) num_workers(1) vector_length(32)
{
int j;
#pragma acc loop gang(static:*)
for (j = 0; j < 32; j++)
n[j]++;
#pragma acc loop gang vector
for (j = 0; j < 1024; j++)
if ((arr[j] % 2) != 0)
arr[j] *= 2;
#pragma acc loop gang(static:*)
for (j = 0; j < 32; j++)
n[j]++;
}
for (i = 0; i < 32; i++)
assert (n[i] == 2);
for (i = 0; i < 1024; i++)
assert (arr[i] == ((i % 2) == 0 ? i : i * 2));
}
void t7()
{
int n = 0;
#pragma acc parallel copy(n) num_gangs(1) num_workers(1) vector_length(32)
{
n++;
}
assert (n == 1);
}
void t8()
{
int arr[1024];
int gangs;
for (gangs = 1; gangs <= 1024; gangs <<= 1)
{
int i;
for (i = 0; i < 1024; i++)
arr[i] = 0;
#pragma acc parallel copy(arr) num_gangs(gangs) num_workers(1) vector_length(32)
{
int j;
#pragma acc loop gang
for (j = 0; j < 1024; j++)
arr[j]++;
}
for (i = 0; i < 1024; i++)
assert (arr[i] == 1);
}
}
void t9()
{
int arr[1024];
int gangs;
for (gangs = 1; gangs <= 1024; gangs <<= 1)
{
int i;
for (i = 0; i < 1024; i++)
arr[i] = 0;
#pragma acc parallel copy(arr) num_gangs(gangs) num_workers(1) vector_length(32)
{
int j;
#pragma acc loop gang
for (j = 0; j < 1024; j++)
if ((j % 3) == 0)
arr[j]++;
else
arr[j] += 2;
}
for (i = 0; i < 1024; i++)
assert (arr[i] == ((i % 3) == 0) ? 1 : 2);
}
}
void t10()
{
int arr[1024];
int gangs;
for (gangs = 1; gangs <= 1024; gangs <<= 1)
{
int i;
for (i = 0; i < 1024; i++)
arr[i] = 0;
#pragma acc parallel copy(arr) num_gangs(gangs) num_workers(1) vector_length(32)
{
int j;
#pragma acc loop gang
for (j = 0; j < 1024; j++)
switch (j % 5)
{
case 0: arr[j] += 1; break;
case 1: arr[j] += 2; break;
case 2: arr[j] += 3; break;
case 3: arr[j] += 4; break;
case 4: arr[j] += 5; break;
default: arr[j] += 99;
}
}
for (i = 0; i < 1024; i++)
assert (arr[i] == (i % 5) + 1);
}
}
void t11()
{
int arr[1024];
int i;
for (i = 0; i < 1024; i++)
arr[i] = 99;
#pragma acc parallel copy(arr) num_gangs(1024) num_workers(1) vector_length(32)
{
int j;
#pragma acc loop gang(static:*)
for (j = 0; j < 1024; j++)
arr[j] = 0;
#pragma acc loop gang(static:*)
for (j = 0; j < 1024; j++)
switch (j % 5)
{
case 0: arr[j] += 1; break;
case 1: arr[j] += 2; break;
case 2: arr[j] += 3; break;
case 3: arr[j] += 4; break;
case 4: arr[j] += 5; break;
default: arr[j] += 99;
}
}
for (i = 0; i < 1024; i++)
assert (arr[i] == (i % 5) + 1);
}
#define NUM_GANGS 4096
void t12()
{
bool fizz[NUM_GANGS], buzz[NUM_GANGS], fizzbuzz[NUM_GANGS];
int i;
#pragma acc parallel copyout(fizz, buzz, fizzbuzz) num_gangs(NUM_GANGS) num_workers(1) vector_length(32)
{
int j;
#pragma acc loop gang(static:*)
for (j = 0; j < NUM_GANGS; j++)
fizz[j] = buzz[j] = fizzbuzz[j] = 0;
#pragma acc loop gang(static:*)
for (j = 0; j < NUM_GANGS; j++)
{
if ((j % 3) == 0 && (j % 5) == 0)
fizzbuzz[j] = 1;
else
{
if ((j % 3) == 0)
fizz[j] = 1;
else if ((j % 5) == 0)
buzz[j] = 1;
}
}
}
for (i = 0; i < NUM_GANGS; i++)
{
assert (fizzbuzz[i] == ((i % 3) == 0 && (i % 5) == 0));
assert (fizz[i] == ((i % 3) == 0 && (i % 5) != 0));
assert (buzz[i] == ((i % 3) != 0 && (i % 5) == 0));
}
}
#undef NUM_GANGS
void t13()
{
int arr[32 * 8], i;
for (i = 0; i < 32 * 8; i++)
arr[i] = 0;
#pragma acc parallel copy(arr) num_gangs(8) num_workers(8) vector_length(32)
{
int j;
#pragma acc loop gang
for (j = 0; j < 32; j++)
{
int k;
#pragma acc loop worker
for (k = 0; k < 8; k++)
arr[j * 8 + k] += j * 8 + k;
}
}
for (i = 0; i < 32 * 8; i++)
assert (arr[i] == i);
}
void t16()
{
int n[32], arr[32 * 32], i;
for (i = 0; i < 32 * 32; i++)
arr[i] = 0;
for (i = 0; i < 32; i++)
n[i] = 0;
#pragma acc parallel copy(n, arr) num_gangs(8) num_workers(16) vector_length(32)
{
int j;
#pragma acc loop gang
for (j = 0; j < 32; j++)
{
int k;
n[j]++;
#pragma acc loop worker
for (k = 0; k < 32; k++)
arr[j * 32 + k]++;
n[j]++;
#pragma acc loop worker
for (k = 0; k < 32; k++)
arr[j * 32 + k]++;
n[j]++;
#pragma acc loop worker
for (k = 0; k < 32; k++)
arr[j * 32 + k]++;
n[j]++;
}
}
for (i = 0; i < 32; i++)
assert (n[i] == 4);
for (i = 0; i < 32 * 32; i++)
assert (arr[i] == 3);
}
void t17()
{
int arr_a[32 * 32], arr_b[32 * 32], i;
int num_workers, num_gangs;
for (num_workers = 1; num_workers <= 32; num_workers <<= 1)
for (num_gangs = 1; num_gangs <= 32; num_gangs <<= 1)
{
for (i = 0; i < 32 * 32; i++)
arr_a[i] = i;
#pragma acc parallel copyin(arr_a) copyout(arr_b) num_gangs(num_gangs) num_workers(num_workers) vector_length(32)
{
int j;
#pragma acc loop gang
for (j = 0; j < 32; j++)
{
int k;
#pragma acc loop worker
for (k = 0; k < 32; k++)
arr_b[j * 32 + (31 - k)] = arr_a[j * 32 + k] * 2;
#pragma acc loop worker
for (k = 0; k < 32; k++)
arr_a[j * 32 + (31 - k)] = arr_b[j * 32 + k] * 2;
#pragma acc loop worker
for (k = 0; k < 32; k++)
arr_b[j * 32 + (31 - k)] = arr_a[j * 32 + k] * 2;
}
}
for (i = 0; i < 32 * 32; i++)
assert (arr_b[i] == (i ^ 31) * 8);
}
}
void t18()
{
int arr_a[32 * 32 * 32], arr_b[32 * 32 * 32], i;
int num_workers, num_gangs;
for (num_workers = 1; num_workers <= 32; num_workers <<= 1)
for (num_gangs = 1; num_gangs <= 32; num_gangs <<= 1)
{
for (i = 0; i < 32 * 32 * 32; i++)
arr_a[i] = i;
#pragma acc parallel copyin(arr_a) copyout(arr_b) num_gangs(num_gangs) num_workers(num_workers) vector_length(32)
{
int j;
#pragma acc loop gang
for (j = 0; j < 32; j++)
{
int k;
#pragma acc loop worker vector
for (k = 0; k < 32 * 32; k++)
arr_b[j * 32 * 32 + (1023 - k)] = arr_a[j * 32 * 32 + k] * 2;
#pragma acc loop worker vector
for (k = 0; k < 32 * 32; k++)
arr_a[j * 32 * 32 + (1023 - k)] = arr_b[j * 32 * 32 + k] * 2;
#pragma acc loop worker vector
for (k = 0; k < 32 * 32; k++)
arr_b[j * 32 * 32 + (1023 - k)] = arr_a[j * 32 * 32 + k] * 2;
}
}
for (i = 0; i < 32 * 32 * 32; i++)
assert (arr_b[i] == (i ^ 1023) * 8);
}
}
void t19()
{
int n[32 * 32], arr_a[32 * 32 * 32], arr_b[32 * 32 * 32], i;
int num_workers, num_gangs;
for (num_workers = 1; num_workers <= 32; num_workers <<= 1)
for (num_gangs = 1; num_gangs <= 32; num_gangs <<= 1)
{
for (i = 0; i < 32 * 32 * 32; i++)
arr_a[i] = i;
for (i = 0; i < 32 * 32; i++)
n[i] = 0;
#pragma acc parallel copy (n) copyin(arr_a) copyout(arr_b) num_gangs(num_gangs) num_workers(num_workers) vector_length(32)
{
int j;
#pragma acc loop gang
for (j = 0; j < 32; j++)
{
int k;
#pragma acc loop worker
for (k = 0; k < 32; k++)
{
int m;
n[j * 32 + k]++;
#pragma acc loop vector
for (m = 0; m < 32; m++)
{
if (((j * 1024 + k * 32 + m) % 2) == 0)
arr_b[j * 1024 + k * 32 + (31 - m)]
= arr_a[j * 1024 + k * 32 + m] * 2;
else
arr_b[j * 1024 + k * 32 + (31 - m)]
= arr_a[j * 1024 + k * 32 + m] * 3;
}
n[j * 32 + k]++;
#pragma acc loop vector
for (m = 0; m < 32; m++)
{
if (((j * 1024 + k * 32 + m) % 3) == 0)
arr_a[j * 1024 + k * 32 + (31 - m)]
= arr_b[j * 1024 + k * 32 + m] * 5;
else
arr_a[j * 1024 + k * 32 + (31 - m)]
= arr_b[j * 1024 + k * 32 + m] * 7;
}
#pragma acc loop vector
for (m = 0; m < 32; m++)
{
if (((j * 1024 + k * 32 + m) % 2) == 0)
arr_b[j * 1024 + k * 32 + (31 - m)]
= arr_a[j * 1024 + k * 32 + m] * 3;
else
arr_b[j * 1024 + k * 32 + (31 - m)]
= arr_a[j * 1024 + k * 32 + m] * 2;
}
}
}
}
for (i = 0; i < 32 * 32; i++)
assert (n[i] == 2);
for (i = 0; i < 32 * 32 * 32; i++)
{
int m = 6 * ((i % 3) == 0 ? 5 : 7);
assert (arr_b[i] == (i ^ 31) * m);
}
}
}
void t20()
{
int w0 = 0;
int w1 = 0;
int w2 = 0;
int w3 = 0;
int w4 = 0;
int w5 = 0;
int w6 = 0;
int w7 = 0;
int i;
#pragma acc parallel copy (w0, w1, w2, w3, w4, w5, w6, w7) num_gangs (1) num_workers (8)
{
int internal = 100;
#pragma acc loop worker
for (i = 0; i < 8; i++)
{
switch (i)
{
case 0: w0 = internal; break;
case 1: w1 = internal; break;
case 2: w2 = internal; break;
case 3: w3 = internal; break;
case 4: w4 = internal; break;
case 5: w5 = internal; break;
case 6: w6 = internal; break;
case 7: w7 = internal; break;
default: break;
}
}
}
if (w0 != 100
|| w1 != 100
|| w2 != 100
|| w3 != 100
|| w4 != 100
|| w5 != 100
|| w6 != 100
|| w7 != 100)
__builtin_abort ();
}
void t21()
{
int arr[32], i;
for (i = 0; i < 32; i++)
arr[i] = 0;
#pragma acc parallel copy(arr) num_gangs(8) num_workers(8) vector_length(32)
{
int j;
#pragma acc loop gang
for (j = 0; j < 32; j++)
arr[j]++;
}
for (i = 0; i < 32; i++)
assert (arr[i] == 1);
}
void t22()
{
int arr[32], i;
for (i = 0; i < 32; i++)
arr[i] = 0;
#pragma acc parallel copy(arr) num_gangs(8) num_workers(8) vector_length(32)
{
int j;
#pragma acc loop gang
for (j = 0; j < 32; j++)
{
#pragma acc atomic
arr[j]++;
}
}
for (i = 0; i < 32; i++)
assert (arr[i] == 1);
}
void t23()
{
int arr[32], i;
for (i = 0; i < 32; i++)
arr[i] = i;
#pragma acc parallel copy(arr) num_gangs(8) num_workers(8) vector_length(32)
{
int j;
#pragma acc loop gang
for (j = 0; j < 32; j++)
if ((arr[j] % 2) != 0)
arr[j]++;
else
arr[j] += 2;
}
for (i = 0; i < 32; i++)
assert (arr[i] == ((i % 2) != 0) ? i + 1 : i + 2);
}
void t24()
{
int arr[32], i;
for (i = 0; i < 32; i++)
arr[i] = i;
#pragma acc parallel copy(arr) num_gangs(8) num_workers(8) vector_length(32)
{
int j;
#pragma acc loop gang
for (j = 0; j < 32; j++)
switch (arr[j] % 5)
{
case 0: arr[j] += 1; break;
case 1: arr[j] += 2; break;
case 2: arr[j] += 3; break;
case 3: arr[j] += 4; break;
case 4: arr[j] += 5; break;
default: arr[j] += 99;
}
}
for (i = 0; i < 32; i++)
assert (arr[i] == i + (i % 5) + 1);
}
void t25()
{
int arr[32 * 32], i;
for (i = 0; i < 32 * 32; i++)
arr[i] = i;
#pragma acc parallel copy(arr) num_gangs(8) num_workers(8) vector_length(32)
{
int j;
#pragma acc loop gang
for (j = 0; j < 32; j++)
{
int k;
#pragma acc loop vector
for (k = 0; k < 32; k++)
{
#pragma acc atomic
arr[j * 32 + k]++;
}
}
}
for (i = 0; i < 32 * 32; i++)
assert (arr[i] == i + 1);
}
#define ACTUAL_GANGS 8
void t27()
{
int n, arr[32], i;
int ondev;
for (i = 0; i < 32; i++)
arr[i] = 0;
n = 0;
#pragma acc parallel copy(n, arr) copyout(ondev) num_gangs(ACTUAL_GANGS) num_workers(8) vector_length(32)
{
int j;
ondev = acc_on_device (acc_device_not_host);
#pragma acc atomic
n++;
#pragma acc loop vector
for (j = 0; j < 32; j++)
{
#pragma acc atomic
arr[j] += 1;
}
#pragma acc atomic
n++;
}
int m = ondev ? ACTUAL_GANGS : 1;
assert (n == m * 2);
for (i = 0; i < 32; i++)
assert (arr[i] == m);
}
#undef ACTUAL_GANGS
#pragma acc routine
float t28_routine ()
{
return 2.71;
}
#define N 32
void t28()
{
float threads[N], v1 = 3.14;
for (int i = 0; i < N; i++)
threads[i] = -1;
#pragma acc parallel num_gangs (1) vector_length (32) copy (v1)
{
float val = t28_routine ();
#pragma acc loop vector
for (int i = 0; i < N; i++)
threads[i] = val + v1*i;
}
for (int i = 0; i < N; i++)
assert (fabs (threads[i] - (t28_routine () + v1*i)) < 0.0001);
}
#undef N
int main()
{
t1();
t2();
t4();
t5();
t7();
t8();
t9();
t10();
t11();
t12();
t13();
t16();
t17();
t18();
t19();
t20();
t21();
t22();
t23();
t24();
t25();
t27();
t28();
return 0;
}
