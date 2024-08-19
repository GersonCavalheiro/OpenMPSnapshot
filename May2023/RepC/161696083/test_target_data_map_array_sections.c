#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "ompvv.h"
#define N 1000
void init_1d(int* a);
void init_2d(int a[N][2]);
void init_3d(int a[N][2][2]);
int test_lower_length_1d() {
OMPVV_INFOMSG("test_lower_length_1d");
int errors = 0;
int a1d[N];
init_1d(a1d);
#pragma omp target data map(from: a1d[1:N - 2])
{
#pragma omp target map(alloc: a1d[1:N - 2]) 
{
for (int i = 1; i < N - 1; ++i)
a1d[i] = 1;
} 
} 
for (int i = 0; i < N; ++i) {
if (i == 0 || i == N - 1){
OMPVV_TEST_AND_SET_VERBOSE(errors, (a1d[i] != 0));
}
else { 
OMPVV_TEST_AND_SET_VERBOSE(errors, (a1d[i] != 1)); 
}
}
return errors;
}
int test_lower_length_2d() {
OMPVV_INFOMSG("test_lower_length_2d");
int errors = 0;
int a2d[N][2];
init_2d(a2d);
#pragma omp target data map(from: a2d[1:N - 2][0:2])
{
#pragma omp target map(alloc: a2d[1:N - 2][0:2]) 
{
for (int i = 1; i < N - 1; ++i) {
a2d[i][0] = 1;
a2d[i][1] = 1;
}
} 
} 
for (int i = 0; i < N; ++i) {
if (i == 0 || i == N - 1){
OMPVV_TEST_AND_SET_VERBOSE(errors, a2d[i][0] != 0 && a2d[i][1] != 0);
} 
else {
OMPVV_TEST_AND_SET_VERBOSE(errors, a2d[i][0] != 1 && a2d[i][1] != 1);
}
}
return errors;
}
int test_lower_length_3d() {
OMPVV_INFOMSG("test_lower_length_3d");
int errors = 0;
int a3d[N][2][2];
init_3d(a3d);
int a3d2[N][2][2];
init_3d(a3d2);
#pragma omp target data map(from: a3d[1:N - 2][0:2][0:2])  map(from: a3d2[0:N][0:2][0:2])
{
#pragma omp target map(alloc: a3d[1:N - 2][0:2][0:2] ,a3d2[0:N][0:2][0:2]) 
{
for (int i = 0; i < N; ++i) {
for (int j = 0; j < 2; ++j) {
if (i > 0 && i < N - 1) {
a3d[i][j][0] = 1;
a3d[i][j][1] = 1;
}
a3d2[i][j][0] = 1;
a3d2[i][j][1] = 1;
}
}
} 
} 
for (int i = 0; i < N; ++i) {
for (int j = 0; j < 2; ++j) {
if (i == 0 || i == N - 1) {
OMPVV_TEST_AND_SET_VERBOSE(errors, a3d[i][j][0] != 0 && a3d[i][j][1] != 0);
} 
else {
OMPVV_TEST_AND_SET_VERBOSE(errors, a3d[i][j][0] != 1 && a3d[i][j][1] != 1);
}
OMPVV_TEST_AND_SET_VERBOSE(errors, a3d2[i][j][0] != 1 && a3d2[i][j][1] != 1);
}
}
return errors;
}
int test_length_1d() {
OMPVV_INFOMSG("test_length_1d");
int errors = 0;
int a1d[N];
init_1d(a1d);
#pragma omp target data map(from: a1d[:N - 2]) 
{
#pragma omp target map(alloc: a1d[:N - 2]) 
{
for (int i = 0; i < N - 2; ++i)
a1d[i] = 1;
} 
} 
for (int i = 0; i < N - 2; ++i)
OMPVV_TEST_AND_SET_VERBOSE(errors, a1d[i] != 1);
OMPVV_TEST_AND_SET_VERBOSE(errors, a1d[N - 2] != 0);
OMPVV_TEST_AND_SET_VERBOSE(errors, a1d[N - 1] != 0);
return errors;
}
int test_length_2d() {
OMPVV_INFOMSG("test_length_2d");
int errors = 0;
int a2d[N][2];
init_2d(a2d);
#pragma omp target data map(from: a2d[:N - 2][:2])
{
#pragma omp target map(alloc: a2d[:N - 2][:2]) 
{
for (int i = 0; i < N - 2; ++i) {
a2d[i][0] = 1;
a2d[i][1] = 1;
}
} 
} 
for (int i = 0; i < N - 2; ++i)
OMPVV_TEST_AND_SET_VERBOSE(errors, a2d[i][0] != 1 && a2d[i][1] != 1);
OMPVV_TEST_AND_SET_VERBOSE(errors, a2d[N - 2][0] != 0 && a2d[N - 2][1] != 0);
OMPVV_TEST_AND_SET_VERBOSE(errors, a2d[N - 1][0] != 0 && a2d[N - 1][1] != 0);
return errors;
}
int test_length_3d() {
OMPVV_INFOMSG("test_length_3d");
int errors = 0;
int a3d[N][2][2];
init_3d(a3d);
int a3d2[N][2][2];
init_3d(a3d2);
#pragma omp target data map(from: a3d[:N - 2][:2][:2])   map(from: a3d2[:N][:2][:2])
{
#pragma omp target map(alloc: a3d[:N - 2][:2][:2], a3d2[:N][:2][:2]) 
{
for (int i = 0; i < N; ++i) {
for (int j = 0; j < 2; ++j) {
if (i < N - 2) {
a3d[i][j][0] = 1;
a3d[i][j][1] = 1;
}
a3d2[i][j][0] = 1;
a3d2[i][j][1] = 1;
}
}
} 
} 
for (int i = 0; i < N; ++i) {
for (int j = 0; j < 2; ++j) {
if (i >= N - 2) {
OMPVV_TEST_AND_SET_VERBOSE(errors, a3d[i][j][0] != 0 && a3d[i][j][1] != 0);
} 
else  
OMPVV_TEST_AND_SET_VERBOSE(errors, a3d[i][j][0] != 1 && a3d[i][j][1] != 1)
OMPVV_TEST_AND_SET_VERBOSE(errors, a3d2[i][j][0] != 1 && a3d2[i][j][1] != 1);
}
}
return errors;
}
int test_lower_1d() {
OMPVV_INFOMSG("test_lower_1d");
int errors = 0;
int a1d[N];
init_1d(a1d);
#pragma omp target data map(from: a1d[1:])
{
#pragma omp target map(alloc: a1d[1:]) 
{
for (int i = 1; i < N; ++i)
a1d[i] = 1;
} 
} 
for (int i = 0; i < N; ++i) {
if (i == 0) {
OMPVV_TEST_AND_SET_VERBOSE(errors, a1d[i] != 0);
}
else
OMPVV_TEST_AND_SET_VERBOSE(errors, a1d[i] != 1);
}
return errors;
}
int test_lower_2d() {
OMPVV_INFOMSG("test_lower_2d");
int errors = 0;
int a2d[N][2];
init_2d(a2d);
#pragma omp target data map(from: a2d[1:][0:])
{
#pragma omp target map(alloc: a2d[1:][0:]) 
{
for (int i = 1; i < N; ++i) {
a2d[i][0] = 1;
a2d[i][1] = 1;
}
} 
} 
for (int i = 0; i < N; ++i) {
if (i == 0) {
OMPVV_TEST_AND_SET_VERBOSE(errors, a2d[i][0] != 0 && a2d[i][1] != 0);
}
else
OMPVV_TEST_AND_SET_VERBOSE(errors, a2d[i][0] != 1 && a2d[i][1] != 1)
}
return errors;
}
int test_lower_3d() {
OMPVV_INFOMSG("test_lower_3d");
int errors = 0;
int a3d[N][2][2];
init_3d(a3d);
int a3d2[N][2][2];
init_3d(a3d2);
#pragma omp target data map(from: a3d[1:][0:][0:])   map(from: a3d2[0:][0:][0:])
{
#pragma omp target map(alloc: a3d[1:][0:][0:], a3d2[0:][0:][0:]) 
{
for (int i = 0; i < N; ++i) {
for (int j = 0; j < 2; ++j) {
if (i > 0) {
a3d[i][j][0] = 1;
a3d[i][j][1] = 1;
}
a3d2[i][j][0] = 1;
a3d2[i][j][1] = 1;
}
}
} 
} 
for (int i = 0; i < N; ++i) {
for (int j = 0; j < 2; ++j) {
if (i == 0) {
OMPVV_TEST_AND_SET_VERBOSE(errors, a3d[i][j][0] != 0 && a3d[i][j][1] != 0);
} 
else {
OMPVV_TEST_AND_SET_VERBOSE(errors, a3d[i][j][0] != 1 && a3d[i][j][1] != 1);
}
OMPVV_TEST_AND_SET_VERBOSE(errors, a3d2[i][j][0] != 1 && a3d2[i][j][1] != 1);
}
}
return errors;
}
int main() {
int errors = 0;
OMPVV_TEST_OFFLOADING;
OMPVV_TEST_AND_SET_VERBOSE(errors, test_lower_length_1d());
OMPVV_TEST_AND_SET_VERBOSE(errors, test_lower_length_2d());
OMPVV_TEST_AND_SET_VERBOSE(errors, test_lower_length_3d());
OMPVV_TEST_AND_SET_VERBOSE(errors, test_length_1d());
OMPVV_TEST_AND_SET_VERBOSE(errors, test_length_2d());
OMPVV_TEST_AND_SET_VERBOSE(errors, test_length_3d());
OMPVV_TEST_AND_SET_VERBOSE(errors, test_lower_1d());
OMPVV_TEST_AND_SET_VERBOSE(errors, test_lower_2d());
OMPVV_TEST_AND_SET_VERBOSE(errors, test_lower_3d());
OMPVV_REPORT_AND_RETURN(errors);
}
void init_1d(int* a) {
for (int i = 0; i < N; ++i)
a[i] = 0;
}
void init_2d(int a[N][2]) {
for (int i = 0; i < N; ++i) {
a[i][0] = 0;
a[i][1] = 0;
}
}
void init_3d(int a[N][2][2]) {
for (int i = 0; i < N; ++i)
for (int j = 0; j < 2; ++j) {
a[i][j][0] = 0;
a[i][j][1] = 0;
}
}
