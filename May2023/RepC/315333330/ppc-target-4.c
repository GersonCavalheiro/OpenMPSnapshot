#ifndef SIZE
#define SIZE 1024
#endif
#ifdef __ALTIVEC__
#error "__ALTIVEC__ should not be defined."
#endif
#ifdef __VSX__
#error "__VSX__ should not be defined."
#endif
#pragma GCC target("vsx")
#include <altivec.h>
#pragma GCC reset_options
#pragma GCC push_options
#pragma GCC target("altivec,no-vsx")
#ifndef __ALTIVEC__
#error "__ALTIVEC__ should be defined."
#endif
#ifdef __VSX__
#error "__VSX__ should not be defined."
#endif
void
av_add (vector float *a, vector float *b, vector float *c)
{
unsigned long i;
unsigned long n = SIZE / 4;
for (i = 0; i < n; i++)
a[i] = vec_add (b[i], c[i]);
}
#pragma GCC target("vsx")
#ifndef __ALTIVEC__
#error "__ALTIVEC__ should be defined."
#endif
#ifndef __VSX__
#error "__VSX__ should be defined."
#endif
void
vsx_add (vector float *a, vector float *b, vector float *c)
{
unsigned long i;
unsigned long n = SIZE / 4;
for (i = 0; i < n; i++)
a[i] = vec_add (b[i], c[i]);
}
#pragma GCC pop_options
#pragma GCC target("no-vsx,no-altivec")
#ifdef __ALTIVEC__
#error "__ALTIVEC__ should not be defined."
#endif
#ifdef __VSX__
#error "__VSX__ should not be defined."
#endif
void
norm_add (float *a, float *b, float *c)
{
unsigned long i;
for (i = 0; i < SIZE; i++)
a[i] = b[i] + c[i];
}
