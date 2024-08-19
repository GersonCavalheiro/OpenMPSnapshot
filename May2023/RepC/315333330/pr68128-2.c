extern int omp_get_thread_num (void);
extern int omp_get_ancestor_thread_num (int);
void b1 (int, int);
int
f1 (void)
{
int a1 = 1;
unsigned char a2 = 2;
unsigned long a3 = 3;
long long a4 = 4;
short a5 = 5;
char a6 = 6;
#pragma omp parallel shared (a1, a2, a3)
{
if (omp_get_thread_num () == 0)
{
a1 = a2;
a4 = a5;
}
b1 (a2, a6);
#pragma omp barrier
if (omp_get_thread_num () == 1)
{
a1 += a3;
a4 += a6;
}
}
return a1 + a2 + a3 + a4 + a5 + a6;
}
struct S { int a, b; };
void b2 (int *, int *, int, int, struct S, struct S);
void
f2 (void)
{
struct S a7 = { 7, 7 }, a8 = { 8, 8 };
int a9 = 9, a10 = 10;
short a11[2] = { 11, 11 };
char a12[1] = { 12 };
#pragma omp parallel shared (a7, a9, a11)
{
b2 (&a9, &a10, a11[1], a12[0], a7, a8);
}
}
void b3 (_Complex float, _Complex float);
_Complex float
f3 (void)
{
_Complex float a13 = 13.0f, a14 = 14.0f, a15 = 15.0f, a16 = 16.0f;
#pragma omp parallel shared (a13, a15)
{
#pragma omp parallel shared (a14)
{
if (omp_get_thread_num () == 0 && omp_get_ancestor_thread_num (1) == 1)
__imag__ a13 = __real__ a15;
else if (omp_get_thread_num () == 1 && omp_get_ancestor_thread_num (1) == 0)
__real__ a14 = __imag__ a16;
b3 (a15, a16);
}
}
return a13 + a14 + a15 + a16;
}
int
f4 (void)
{
int a17 = 17, a18 = 18, a19 = 19, a20 = 20, a21 = 21, a22 = 22, a23 = 23, a24 = 0, a25 = 0, a26 = 0;
int i;
#pragma omp task shared (a17)
b1 (a17, a18);
b1 (a17, a18);
#pragma omp taskwait
#pragma omp parallel shared (a19)
{
#pragma omp task shared (a19)
{
a19 = 1;
}
#pragma omp task shared (a20)
a20 = a21;
#pragma omp for firstprivate (a25) lastprivate (a22) linear (a23:2) reduction (+:a24) private (a26)
for (i = 0; i < 10; i++)
{
a26 = i;
a22 = a26 + 7;
a23 += 2;
a24 += i;
a25++;
}
}
return a22 + a23 + a24 + a25 + a26;
}
void
f5 (void)
{
int a27 = 27, a28 = 28, a29 = 29, a30 = 30;
#pragma omp target data map (tofrom: a27, a28)
{
#pragma omp target map (tofrom: a27)
a27++;
#pragma omp parallel shared (a27, a28)
{
#pragma omp critical
{
#pragma omp target update to (a27)
#pragma omp target map (always, from: a28) private (a30)
{
a28++;
a29++;
a30 = a29;
}
}
#pragma omp barrier
b1 (a27, a28);
}
}
}
int
f6 (void)
{
int a31 = 31, a32 = 32, a33 = 33, a34 = 34;
#pragma omp parallel
{
#pragma omp sections
{
#pragma omp section
{
a31 = 9;
}
#pragma omp section
{
int i = 10;
__builtin_memcpy (&a32, &i, sizeof (int));
}
}
#pragma omp single
a33 = 11;
#pragma omp atomic
a34++;
}
return a31 + a32 + a33 + a34;
}
