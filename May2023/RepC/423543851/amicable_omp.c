#include <stdio.h>
#include <unistd.h>
#include <math.h>
#include <omp.h>
#define N 10000000
#define THREADS 32
#define PACK_SIZE 1000
int sum_all = 0;
int vsote[N+1];
double start, end;
void vsota_staticno_enakomerno() {
#pragma omp parallel for schedule(static, N/THREADS)
for(int i = 0; i < N; i++) {
int sum = 1;
int koren = sqrt(i);
for(int j = 2; j <= koren; j++) {
if(i % j == 0){
if(j == (i / j)) {
sum += j;
} else {
sum += (j + i / j);
}
}
}
vsote[i] = sum;
}
}
void vsota_staticno_krozno() {
#pragma omp parallel for schedule(static, PACK_SIZE)
for(int i = 0; i < N; i++) {
int sum = 1;
int koren = sqrt(i);
for(int j = 2; j <= koren; j++) {
if(i % j == 0){
if(j == (i / j)) {
sum += j;
} else {
sum += (j + i / j);
}
}
}
vsote[i] = sum;
}
}
void vsota_dinamicno() {
#pragma omp parallel for schedule(dynamic, PACK_SIZE)
for(int i = 0; i < N; i++) {
int sum = 1;
int koren = sqrt(i);
for(int j = 2; j <= koren; j++) {
if(i % j == 0){
if(j == (i / j)) {
sum += j;
} else {
sum += (j + i / j);
}
}
}
vsote[i] = sum;
}
}
void pari() {
#pragma omg parallel for schedule(static, N/THREADS)
for(int i = 0; i < N; i++) {
int a = vsote[i];
int b; 
if(a <= N) {
b = vsote[a];
if(i == b && a != b) {
#pragma omp critical
{
sum_all += (a + b);
vsote[a] = -1;
}
}
}
}
}
int main() {
start = omp_get_wtime();
vsota_staticno_enakomerno();
end = omp_get_wtime(); 
printf("Staticno enakomerno:  %f s\n", end - start);
start = omp_get_wtime();
vsota_staticno_krozno();
end = omp_get_wtime(); 
printf("Staticno krozno (Np=%d):      %f s\n", PACK_SIZE, end - start);
start = omp_get_wtime();
vsota_dinamicno();
end = omp_get_wtime();
printf("Dinamicno (Np=%d):  %f s\n", PACK_SIZE, end - start);
pari();
printf("Vsota = %d\n", sum_all);
return 0;
}
