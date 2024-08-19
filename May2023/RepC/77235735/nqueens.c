#include <stdlib.h>
#include <stdio.h>
#include <memory.h>
#include <alloca.h>
#include "bots.h"
#include "app-desc.h"
#include <omp.h>
static int solutions[] = {
1,
0,
0,
2,
10, 
4,
40,
92,
352,
724, 
2680,
14200,
73712,
365596,
};
#define MAX_SOLUTIONS sizeof(solutions)/sizeof(int)
#ifdef FORCE_TIED_TASKS
int mycount=0;
#pragma omp threadprivate(mycount)
#endif
int total_count;
int ok(int n, char *a)
{
int i, j;
char p, q;
for (i = 0; i < n; i++) {
p = a[i];
for (j = i + 1; j < n; j++) {
q = a[j];
if (q == p || q == p - (j - i) || q == p + (j - i))
return 0;
}
}
return 1;
}
#ifndef FORCE_TIED_TASKS
void nqueens_ser (int n, int j, char *a, int *solutions)
#else
void nqueens_ser (int n, int j, char *a)
#endif
{
#ifndef FORCE_TIED_TASKS
int res;
#endif
int i;
if (n == j) {
#ifndef FORCE_TIED_TASKS
*solutions = 1;
#else
mycount++;
#endif
return;
}
#ifndef FORCE_TIED_TASKS
*solutions = 0;
#endif
for (i = 0; i < n; i++) {
{
a[j] = (char) i;
if (ok(j + 1, a)) {
#ifndef FORCE_TIED_TASKS
nqueens_ser(n, j + 1, a,&res);
*solutions += res;
#else
nqueens_ser(n, j + 1, a);
#endif
}
}
}
}
#if defined(IF_CUTOFF)
#ifndef FORCE_TIED_TASKS
void nqueens(int n, int j, char *a, int *solutions, int depth)
#else
void nqueens(int n, int j, char *a, int depth)
#endif
{
#ifndef FORCE_TIED_TASKS
int *csols;
#endif
int i;
if (n == j) {
#ifndef FORCE_TIED_TASKS
*solutions = 1;
#else
mycount++;
#endif
return;
}
#ifndef FORCE_TIED_TASKS
*solutions = 0;
csols = (int *)alloca(n*sizeof(int));
memset(csols,0,n*sizeof(int));
#endif
for (i = 0; i < n; i++) {
#pragma omp task untied if(depth < bots_cutoff_value)
{
char * b = (char *)alloca(n * sizeof(char));
memcpy(b, a, j * sizeof(char));
b[j] = (char) i;
if (ok(j + 1, b))
#ifndef FORCE_TIED_TASKS
nqueens(n, j + 1, b,&csols[i],depth+1);
#else
nqueens(n, j + 1, b,depth+1);
#endif
}
}
#pragma omp taskwait
#ifndef FORCE_TIED_TASKS
for ( i = 0; i < n; i++) *solutions += csols[i];
#endif
}
#elif defined(FINAL_CUTOFF)
#ifndef FORCE_TIED_TASKS
void nqueens(int n, int j, char *a, int *solutions, int depth)
#else
void nqueens(int n, int j, char *a, int depth)
#endif
{
#ifndef FORCE_TIED_TASKS
int *csols;
#endif
int i;
if (n == j) {
#ifndef FORCE_TIED_TASKS
*solutions += 1;
#else
mycount++;
#endif
return;
}
#ifndef FORCE_TIED_TASKS
char final = omp_in_final();
if ( !final ) {
*solutions = 0;
csols = (int *)alloca(n*sizeof(int));
memset(csols,0,n*sizeof(int));
}
#endif
for (i = 0; i < n; i++) {
#pragma omp task untied final(depth+1 >= bots_cutoff_value) mergeable
{
char *b;
int *sol;
if ( omp_in_final() && depth+1 > bots_cutoff_value ) {
b = a;
#ifndef FORCE_TIED_TASKS
sol = solutions;
#endif
} else {
b = (char *)alloca(n * sizeof(char));
memcpy(b, a, j * sizeof(char));
#ifndef FORCE_TIED_TASKS
sol = &csols[i];
#endif
} 
b[j] = i;
if (ok(j + 1, b))
#ifndef FORCE_TIED_TASKS
nqueens(n, j + 1, b,sol,depth+1);
#else
nqueens(n, j + 1, b,depth+1);
#endif
}
}
#pragma omp taskwait
#ifndef FORCE_TIED_TASKS
if ( !final ) {
for ( i = 0; i < n; i++) *solutions += csols[i];
}
#endif
}
#elif defined(MANUAL_CUTOFF)
#ifndef FORCE_TIED_TASKS
void nqueens(int n, int j, char *a, int *solutions, int depth)
#else
void nqueens(int n, int j, char *a, int depth)
#endif
{
#ifndef FORCE_TIED_TASKS
int *csols;
#endif
int i;
if (n == j) {
#ifndef FORCE_TIED_TASKS
*solutions = 1;
#else
mycount++;
#endif
return;
}
#ifndef FORCE_TIED_TASKS
*solutions = 0;
csols = (int *)alloca(n*sizeof(int));
memset(csols,0,n*sizeof(int));
#endif
for (i = 0; i < n; i++) {
if ( depth < bots_cutoff_value ) {
#pragma omp task untied
{
char * b = (char *)alloca(n * sizeof(char));
memcpy(b, a, j * sizeof(char));
b[j] = (char) i;
if (ok(j + 1, b))
#ifndef FORCE_TIED_TASKS
nqueens(n, j + 1, b,&csols[i],depth+1);
#else
nqueens(n, j + 1, b,depth+1);
#endif
}
} else {
a[j] = (char) i;
if (ok(j + 1, a))
#ifndef FORCE_TIED_TASKS
nqueens_ser(n, j + 1, a,&csols[i]);
#else
nqueens_ser(n, j + 1, a);
#endif
}
}
#pragma omp taskwait
#ifndef FORCE_TIED_TASKS
for ( i = 0; i < n; i++) *solutions += csols[i];
#endif
}
#else 
#ifndef FORCE_TIED_TASKS
void nqueens(int n, int j, char *a, int *solutions, int depth)
#else
void nqueens(int n, int j, char *a, int depth)
#endif
{
#ifndef FORCE_TIED_TASKS
int *csols;
#endif
int i;
if (n == j) {
#ifndef FORCE_TIED_TASKS
*solutions = 1;
#else
mycount++;
#endif
return;
}
#ifndef FORCE_TIED_TASKS
*solutions = 0;
csols = (int *)alloca(n*sizeof(int));
memset(csols,0,n*sizeof(int));
#endif
for (i = 0; i < n; i++) {
#pragma omp task untied
{
char * b = (char *)alloca(n * sizeof(char));
memcpy(b, a, j * sizeof(char));
b[j] = (char) i;
if (ok(j + 1, b))
#ifndef FORCE_TIED_TASKS
nqueens(n, j + 1, b,&csols[i],depth); 
#else
nqueens(n, j + 1, b,depth); 
#endif
}
}
#pragma omp taskwait
#ifndef FORCE_TIED_TASKS
for ( i = 0; i < n; i++) *solutions += csols[i];
#endif
}
#endif
void find_queens (int size)
{
total_count=0;
bots_message("Computing N-Queens algorithm (n=%d) ", size);
#pragma omp parallel
{
#pragma omp single
{
char *a;
a = (char *)alloca(size * sizeof(char));
#ifndef FORCE_TIED_TASKS
nqueens(size, 0, a, &total_count,0);
#else
nqueens(size, 0, a, 0);
#endif
}
#ifdef FORCE_TIED_TASKS
#pragma omp atomic
total_count += mycount;
#endif
}
bots_message(" completed!\n");
}
int verify_queens (int size)
{
if ( size > MAX_SOLUTIONS ) return BOTS_RESULT_NA;
if ( total_count == solutions[size-1]) return BOTS_RESULT_SUCCESSFUL;
return BOTS_RESULT_UNSUCCESSFUL;
}
