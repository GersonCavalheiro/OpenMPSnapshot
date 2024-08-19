#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include "polybench/polybench.h"
#include "polybench/jacobi-2d-imper.h"
static void init_array(int n,double A[500 + 0][500 + 0],double B[500 + 0][500 + 0])
{
{
int c2;
int c1;
if (n >= 1) {
#pragma omp parallel for private(c2)
for (c1 = 0; c1 <= n + -1; c1++) {
for (c2 = 0; c2 <= n + -1; c2++) {
A[c1][c2] = (((double )c1) * (c2 + 2) + 2) / n;
B[c1][c2] = (((double )c1) * (c2 + 3) + 3) / n;
}
}
}
}
}
static void print_array(int n,double A[500 + 0][500 + 0])
{
int i;
int j;
for (i = 0; i < n; i++) 
for (j = 0; j < n; j++) {
fprintf(stderr,"%0.2lf ",A[i][j]);
if ((i * n + j) % 20 == 0) 
fprintf(stderr,"\n");
}
fprintf(stderr,"\n");
}
static void kernel_jacobi_2d_imper(int tsteps,int n,double A[500 + 0][500 + 0],double B[500 + 0][500 + 0])
{
{
int c2;
int c1;
int c0;
for (c2 = 1; c2 <= 498; c2++) {
B[1][c2] = 0.2 * (A[1][c2] + A[1][c2 - 1] + A[1][1 + c2] + A[1 + 1][c2] + A[1 - 1][c2]);
}
for (c0 = 2; c0 <= 525; c0++) {
if (c0 <= 28) {
if ((2 * c0 + 1) % 3 == 0) {
for (c2 = ((2 * c0 + 1) * 3 < 0?-(-(2 * c0 + 1) / 3) : ((3 < 0?(-(2 * c0 + 1) + - 3 - 1) / - 3 : (2 * c0 + 1 + 3 - 1) / 3))); c2 <= (((2 * c0 + 1492) * 3 < 0?((3 < 0?-((-(2 * c0 + 1492) + 3 + 1) / 3) : -((-(2 * c0 + 1492) + 3 - 1) / 3))) : (2 * c0 + 1492) / 3)); c2++) {
B[1][(-2 * c0 + 3 * c2 + 2) / 3] = 0.2 * (A[1][(-2 * c0 + 3 * c2 + 2) / 3] + A[1][(-2 * c0 + 3 * c2 + 2) / 3 - 1] + A[1][1 + (-2 * c0 + 3 * c2 + 2) / 3] + A[1 + 1][(-2 * c0 + 3 * c2 + 2) / 3] + A[1 - 1][(-2 * c0 + 3 * c2 + 2) / 3]);
}
}
}
#pragma omp parallel for private(c2)
for (c1 = ((((2 * c0 + 2) * 3 < 0?-(-(2 * c0 + 2) / 3) : ((3 < 0?(-(2 * c0 + 2) + - 3 - 1) / - 3 : (2 * c0 + 2 + 3 - 1) / 3)))) > c0 + -9?(((2 * c0 + 2) * 3 < 0?-(-(2 * c0 + 2) / 3) : ((3 < 0?(-(2 * c0 + 2) + - 3 - 1) / - 3 : (2 * c0 + 2 + 3 - 1) / 3)))) : c0 + -9); c1 <= (((((2 * c0 + 498) * 3 < 0?((3 < 0?-((-(2 * c0 + 498) + 3 + 1) / 3) : -((-(2 * c0 + 498) + 3 - 1) / 3))) : (2 * c0 + 498) / 3)) < c0?(((2 * c0 + 498) * 3 < 0?((3 < 0?-((-(2 * c0 + 498) + 3 + 1) / 3) : -((-(2 * c0 + 498) + 3 - 1) / 3))) : (2 * c0 + 498) / 3)) : c0)); c1++) {
B[-2 * c0 + 3 * c1][1] = 0.2 * (A[-2 * c0 + 3 * c1][1] + A[-2 * c0 + 3 * c1][1 - 1] + A[-2 * c0 + 3 * c1][1 + 1] + A[1 + (-2 * c0 + 3 * c1)][1] + A[-2 * c0 + 3 * c1 - 1][1]);
for (c2 = 2 * c0 + -2 * c1 + 2; c2 <= 2 * c0 + -2 * c1 + 498; c2++) {
A[-2 * c0 + 3 * c1 + -1][-2 * c0 + 2 * c1 + c2 + -1] = B[-2 * c0 + 3 * c1 + -1][-2 * c0 + 2 * c1 + c2 + -1];
B[-2 * c0 + 3 * c1][-2 * c0 + 2 * c1 + c2] = 0.2 * (A[-2 * c0 + 3 * c1][-2 * c0 + 2 * c1 + c2] + A[-2 * c0 + 3 * c1][-2 * c0 + 2 * c1 + c2 - 1] + A[-2 * c0 + 3 * c1][1 + (-2 * c0 + 2 * c1 + c2)] + A[1 + (-2 * c0 + 3 * c1)][-2 * c0 + 2 * c1 + c2] + A[-2 * c0 + 3 * c1 - 1][-2 * c0 + 2 * c1 + c2]);
}
A[-2 * c0 + 3 * c1 + -1][498] = B[-2 * c0 + 3 * c1 + -1][498];
}
if (c0 >= 499) {
if ((2 * c0 + 1) % 3 == 0) {
for (c2 = ((2 * c0 + -992) * 3 < 0?-(-(2 * c0 + -992) / 3) : ((3 < 0?(-(2 * c0 + -992) + - 3 - 1) / - 3 : (2 * c0 + -992 + 3 - 1) / 3))); c2 <= (((2 * c0 + 499) * 3 < 0?((3 < 0?-((-(2 * c0 + 499) + 3 + 1) / 3) : -((-(2 * c0 + 499) + 3 - 1) / 3))) : (2 * c0 + 499) / 3)); c2++) {
A[498][(-2 * c0 + 3 * c2 + 995) / 3] = B[498][(-2 * c0 + 3 * c2 + 995) / 3];
}
}
}
}
for (c2 = 20; c2 <= 517; c2++) {
A[498][c2 + -19] = B[498][c2 + -19];
}
}
}
int main(int argc,char **argv)
{
int n = 500;
int tsteps = 10;
double (*A)[500 + 0][500 + 0];
A = ((double (*)[500 + 0][500 + 0])(polybench_alloc_data(((500 + 0) * (500 + 0)),(sizeof(double )))));
;
double (*B)[500 + 0][500 + 0];
B = ((double (*)[500 + 0][500 + 0])(polybench_alloc_data(((500 + 0) * (500 + 0)),(sizeof(double )))));
;
init_array(n, *A, *B);
polybench_timer_start();
;
kernel_jacobi_2d_imper(tsteps,n, *A, *B);
polybench_timer_stop();
;
polybench_timer_print();
;
if (argc > 42 && !strcmp(argv[0],"")) 
print_array(n, *A);
free(((void *)A));
;
free(((void *)B));
;
return 0;
}
